"""Scientific isolation and observer-invariance tests for revision 4."""

import logging
import json

import pytest
import torch

from inverpinn.data.development_only import DEVELOPMENT_IDS, development_path, development_truth
from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.training.objective_diagnostics import ObjectiveAudit, component_gradients
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings, fit_sparse_source
from inverpinn.training.development_candidates import fit_audited_baseline


@pytest.mark.parametrize("identifier", ["heldout_001", "heldout_030", "development_011", "../heldout_001", "development_000"])
def test_heldout_guard_rejects_before_io(tmp_path, identifier):
    with pytest.raises(ValueError, match="ONLY"):
        development_path(tmp_path, identifier)


def test_allowlist_and_symlink_guard(tmp_path):
    assert len(DEVELOPMENT_IDS) == 10
    for identifier in DEVELOPMENT_IDS:
        assert development_path(tmp_path, identifier).name == identifier
    case = tmp_path/"datasets"/"development_001"
    case.mkdir(parents=True)
    (case/"escape").symlink_to(tmp_path/"datasets"/"heldout_001")
    with pytest.raises(ValueError, match="escapes"):
        development_path(tmp_path, "development_001", "escape/source.json")
    with pytest.raises(ValueError, match="escapes"):
        development_path(tmp_path, "development_001", "../development_002/truth")


def test_truth_barrier(tmp_path):
    run = tmp_path/"run"
    run.mkdir()
    (run/"attempt.json").write_text(json.dumps(dict(training_terminated=False)))
    with pytest.raises(RuntimeError, match="terminated"):
        development_truth(tmp_path, "development_001", run)


def small_settings():
    return TrainingSettings(8, 2, 3, .001, 4, 8, 4, 2, 10., 1., 1., 1., .5, .5, .5)


def test_audit_does_not_change_optimizer_rng_or_gradients():
    data = SparseTrainingData(torch.tensor([[.2, .3], [.6, .7]]), torch.tensor([0., .8]),
                              torch.tensor([[0., 0.], [.01, .03]]), .35, -.15, .005)
    settings = small_settings()
    plain = fit_sparse_source(data, settings, .08, 42, logging.getLogger("test"))
    observer = ObjectiveAudit(settings.steps, 1)
    audited = fit_audited_baseline(data, settings, .08, 42, logging.getLogger("test"), on_objective=observer)
    for a, b in zip(plain[:2], audited[:2]):
        for key in a.state_dict():
            assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    assert plain[3] == audited[3]
    assert len(observer.gradients) == 3*4
    for row in observer.gradients:
        if row["component"] != "pde":
            assert row["Q_raw_gradient"] == row["x_s_raw_gradient"] == row["y_s_raw_gradient"] == 0


def test_known_gradient_diagnostics_are_signed_and_chain_rule_correct():
    model = ConcentrationMLP([0,0,0], [1,1,1], 4, 1)
    source = TrainableGaussianSource(.5, .5, .5, .08)
    points = torch.tensor([[.3, .4, .5]])
    losses = dict(data=model(points).square().mean(),
                  pde=(source.Q-1).square()+(source.x_s-.1).square()+(source.y_s-.9).square(),
                  initial=model(points).square().mean(), boundary=model(points).square().mean())
    rows, _ = component_gradients(losses, dict(data=10., pde=2., initial=1., boundary=1.), model, source)
    pde = next(r for r in rows if r["component"] == "pde")
    assert pde["Q_physical_gradient"] == pytest.approx(-1.)
    assert pde["Q_weighted_gradient"] == pytest.approx(-2.)
    assert pde["x_s_physical_gradient"] == pytest.approx(.8)
    assert pde["y_s_physical_gradient"] == pytest.approx(-.8)
    assert pde["x_s_raw_norm"] == pytest.approx(.2)
    assert pde["y_s_raw_norm"] == pytest.approx(.2)
    assert pde["Q_jacobian"] == pytest.approx(1-torch.exp(torch.tensor(-.5)).item())
    assert pde["nn_norm"] == 0
    assert all(p.grad is None for p in [*model.parameters(), *source.parameters()])


@pytest.mark.parametrize("constraint", ["positive", "hard"])
def test_positive_concentration_and_finite_derivatives(constraint):
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.physics.autodiff import concentration_derivatives
    torch.manual_seed(15)
    model = ConstrainedConcentrationMLP(1., 8, 2, constraint).double()
    points = torch.rand(50, 3, dtype=torch.float64).requires_grad_()
    assert torch.all(model(points) >= 0)
    derivatives = concentration_derivatives(model(points), points)
    assert all(torch.isfinite(v).all() for v in derivatives.values())
    model(points).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_hard_zero_constraints_and_full_product_derivatives():
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.physics.autodiff import concentration_derivatives
    model = ConstrainedConcentrationMLP(1., 4, 1, "hard").double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    for axis, value in ((0,0.), (0,1.), (1,0.), (1,1.), (2,0.)):
        points = torch.rand(20, 3, dtype=torch.float64)
        points[:, axis] = value
        assert torch.count_nonzero(model(points)) == 0
    z = torch.tensor([[.2,.3,.4], [.6,.7,.8]], dtype=torch.float64, requires_grad=True)
    x,y,t = z.split(1, dim=1)
    a = torch.log(torch.tensor(2., dtype=torch.float64))
    expected = dict(C_t=16*x*(1-x)*y*(1-y)*a, C_x=16*(1-2*x)*y*(1-y)*t*a,
        C_y=16*x*(1-x)*(1-2*y)*t*a, C_xx=-32*y*(1-y)*t*a, C_yy=-32*x*(1-x)*t*a)
    actual = concentration_derivatives(model(z), z)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("parameterization", ["softplus", "exp"])
@pytest.mark.parametrize("Q", [.01, .5, 1.1, 1.5, 5.])
def test_Q_parameterization_range_and_nonzero_gradient(parameterization, Q):
    from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
    source = DiagnosticGaussianSource(.3,.7,Q,.08,parameterization).double()
    assert source.Q.item() == pytest.approx(Q, rel=1e-6)
    gradient = torch.autograd.grad(source.Q, source.raw_Q)[0].item()
    import math
    assert gradient == pytest.approx(Q if parameterization == "exp" else 1-math.exp(-Q), rel=1e-6)
    with torch.no_grad():
        source.raw_Q.fill_(-1000)
        source.raw_x.fill_(1000)
        source.raw_y.fill_(-1000)
    assert source.Q > 0
    assert 0 <= source.x_s <= 1 and 0 <= source.y_s <= 1


def test_nondimensional_residual_equals_scaled_physical_residual():
    from inverpinn.physics.nondimensional import CharacteristicScales
    from inverpinn.physics.autodiff import autograd_pde_residual
    scales = CharacteristicScales(length=3., time=.8, concentration=.88)
    torch.manual_seed(12)
    hat = torch.rand(13,3,dtype=torch.float64,requires_grad=True)
    physical = (hat.detach()*torch.tensor([3.,3.,.8],dtype=torch.float64)).requires_grad_()
    def field(z):
        x,y,t = z.split(1,dim=1)
        return 2*x*x+3*y**3+.7*x*t+.2*t*t
    source = torch.ones(13,1,dtype=torch.float64)*1.1
    physical_r = autograd_pde_residual(field(physical), physical, u=.35,v=-.15,D=.005,S=source)
    c = field(hat*torch.tensor([3.,3.,.8],dtype=torch.float64))/.88
    hat_r = autograd_pde_residual(c, hat, **scales.transport(.35,-.15,.005), S=source*.8/.88)
    torch.testing.assert_close(hat_r, scales.residual(physical_r), rtol=1e-12, atol=1e-12)
    assert scales.loss_factors()["pde"] == pytest.approx((.8/.88)**2)


def test_candidate_identity_is_bitwise_original_and_repeatable():
    from inverpinn.training.development_candidates import fit_candidate
    data = SparseTrainingData(torch.tensor([[.2,.3],[.6,.7]]),torch.tensor([0.,.8]),
                              torch.tensor([[0.,0.],[.01,.03]]),.35,-.15,.005)
    spec = dict(constraint="unconstrained",Q_parameterization="softplus",scales=None)
    a = fit_sparse_source(data,small_settings(),.08,42)
    b = fit_candidate(data,small_settings(),.08,42,spec)
    c = fit_candidate(data,small_settings(),.08,42,spec)
    for other in (b,c):
        for component in (0,1):
            for key in a[component].state_dict():
                assert torch.equal(a[component].state_dict()[key],other[component].state_dict()[key])
        assert a[3] == other[3]


def test_multistart_selection_cannot_receive_truth_and_breaks_ties_stably():
    from inverpinn.training.development_candidates import select_multistart
    assert select_multistart(dict(center=.2,upper=.1)) == "upper"
    assert select_multistart(dict(z=.1,a=.1)) == "a"
    with pytest.raises(ValueError):
        select_multistart(dict(center=dict(objective=.1,localization_error=0.)))


def test_development_hashing_does_not_modify_or_enumerate_other_scenarios(tmp_path):
    from inverpinn.data.development_only import development_hashes
    for identifier in DEVELOPMENT_IDS:
        case = development_path(tmp_path,identifier)
        case.mkdir(parents=True)
        (case/"truth.bin").write_bytes(identifier.encode())
    other = tmp_path/"datasets"/"heldout_001"
    other.mkdir()
    (other/"secret").write_bytes(b"do not open")
    before = development_hashes(tmp_path)
    assert len(before) == 10
    assert all("development" in key for key in before)
    assert development_hashes(tmp_path) == before


def test_global_open_guard_rejects_shared_manifests_and_reference_writes(tmp_path):
    import os
    from inverpinn.data.development_only import validate_benchmark_open
    for relative in ("scenario_plan.json","scenario_manifest.csv","datasets/heldout_001/truth/source.json",
                     "runs/heldout_030/seed_42/metrics.json"):
        with pytest.raises(PermissionError):
            validate_benchmark_open(tmp_path,tmp_path/relative)
    path=tmp_path/"datasets/development_001/truth/source.json"
    validate_benchmark_open(tmp_path,path)
    for mode,flags in (("w",0),("r+",0),(None,os.O_WRONLY),(None,os.O_CREAT)):
        with pytest.raises(PermissionError,match="read-only"):
            validate_benchmark_open(tmp_path,path,mode,flags)


def test_audit_hook_blocks_real_python_and_numpy_opens_in_subprocess(tmp_path):
    import subprocess
    import sys
    script="""
import sys
import numpy as np
from pathlib import Path
from inverpinn.data.development_only import install_development_read_guard
root=Path(sys.argv[1])
install_development_read_guard(root)
for opener in (lambda: (root/'scenario_plan.json').read_text(),
               lambda: np.load(root/'datasets/heldout_001/truth/checkpoint.npz')):
    try:
        opener()
    except PermissionError:
        continue
    raise AssertionError('Open was not denied before file I/O')
"""
    subprocess.run([sys.executable,"-c",script,str(tmp_path)],check=True,capture_output=True)


def test_candidate_initializations_are_predeclared_and_truth_independent():
    import importlib.util
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("diagnosis_runner",root/"scripts/run_model_diagnosis.py")
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config=yaml.safe_load((root/"configs/model_diagnosis.yaml").read_text())
    plan=yaml.safe_load((root/"configs/model_diagnosis_candidates.yaml").read_text())
    first=runner.resolve_candidates(config,plan)
    torch.manual_seed(123456)
    second=runner.resolve_candidates(config,plan)
    assert first == second
    starts={c["name"]:(c["training"]["initial_x"],c["training"]["initial_y"],c["training"]["initial_Q"]) for c in first}
    assert starts["Q11"] == (.5,.5,1.1)
    assert starts["Q15"] == (.5,.5,1.5)
    assert starts["L_UL"] == (.3,.7,.5)
    assert starts["L_UR"] == (.7,.7,.5)
    assert starts["L_LL"] == (.3,.3,.5)
    assert starts["L_LR"] == (.7,.3,.5)


def test_selection_rejects_physical_or_primary_regressions():
    import pandas as pd
    from inverpinn.evaluation.model_diagnosis import select_revised
    baseline=dict(candidate="B0",count=10,successful=10)
    for key in ("localization_error","relative_strength_error","relative_l2","negative_fraction",
                "pde_rmse","bc_rmse","ic_rmse","training_seconds"):
        baseline.update({f"{key}_{s}":1. for s in ("mean","median","max")})
    safe=baseline|dict(candidate="safe",negative_fraction_max=0.,relative_strength_error_median=.5)
    negative=safe|dict(candidate="negative",negative_fraction_max=.001,relative_strength_error_median=.1)
    poor_C=safe|dict(candidate="poor_C",relative_l2_mean=2.,relative_strength_error_median=.01)
    poor_pde=safe|dict(candidate="poor_PDE",pde_rmse_mean=2.,relative_strength_error_median=.001)
    result=select_revised(pd.DataFrame([baseline,safe,negative,poor_C,poor_pde]))
    assert result["selected"] == "safe"
    assert len(result["exclusion_reasons"]["negative"]) > 0
    assert select_revised(pd.DataFrame([baseline,negative,poor_C,poor_pde]))["selected"] is None


def test_hard_scaled_candidate_is_bitwise_reproducible():
    from inverpinn.training.development_candidates import fit_candidate
    data=SparseTrainingData(torch.tensor([[.2,.3],[.6,.7]]),torch.tensor([0.,.8]),
                            torch.tensor([[0.,0.],[.01,.03]]),.35,-.15,.005)
    spec=dict(constraint="hard",Q_parameterization="softplus",scales=dict(length=1.,time=.8,concentration=.88))
    a=fit_candidate(data,small_settings(),.08,42,spec)
    b=fit_candidate(data,small_settings(),.08,42,spec)
    assert a[3] == b[3]
    for index in (0,1):
        for key in a[index].state_dict():
            assert torch.equal(a[index].state_dict()[key],b[index].state_dict()[key])


def test_amplitude_ray_exact_minimum_and_no_physics_no_shrinkage():
    import numpy as np
    from inverpinn.evaluation.model_diagnosis import amplitude_ray_diagnostic
    predicted=np.array([1.,2.,3.])
    observed=2*predicted
    result=amplitude_ray_diagnostic(predicted,observed,7.,10.)
    a=result["joint_objective_scale"]
    def objective(scale):
        return 10*np.mean((scale*predicted-observed)**2)+7*scale**2
    assert objective(a)<objective(a-.001) and objective(a)<objective(a+.001)
    assert 0<result["objective_shrinkage_ratio"]<1
    exact=amplitude_ray_diagnostic(predicted,observed,0.,10.)
    assert exact["joint_objective_scale"] == exact["data_only_scale"] == 2.
    assert exact["objective_shrinkage_ratio"] == 1.
