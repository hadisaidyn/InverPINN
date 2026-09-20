"""Truth isolation, exact amplitude mathematics and actual block-update tests."""

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar, least_squares
import torch
import yaml

from inverpinn.data.development_v3 import IDS, case_path, check_open, development_plan, previous_plans
from inverpinn.data.fresh_blind import REVISED_HASH
from inverpinn.data.synthetic_reference import ROOT, sha256
from inverpinn.physics.profiled_source import ProfiledGaussianSource, profile_Q
from inverpinn.training.development_candidates import fit_candidate
from inverpinn.training.hybrid_source import observation_profile
from inverpinn.training.inverse_formulation import GradientPathway, activate_block, fit_profiled
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings


def recipe():
    frozen=yaml.safe_load((ROOT/"configs/B_revised.yaml").read_text())
    training=copy.deepcopy(frozen["inference"]["training"])
    training.update(width=8,depth=2,steps=4,data_batch=6,collocation_batch=12,initial_batch=4,boundary_per_edge=2)
    return TrainingSettings.from_mapping(training),frozen["inference"]["formulation"]


def sparse():
    return SparseTrainingData(torch.tensor([[.2,.4],[.6,.7]]),torch.tensor([0.,.4,.8]),torch.tensor([[0.,0.],[.1,.2],[.2,.3]]),.35,-.15,.005)


@pytest.mark.parametrize("weighted",[False,True])
def test_exact_profile_matches_scalar_optimizer(weighted):
    rng=np.random.default_rng(739)
    g=torch.tensor(rng.uniform(.01,1,(200,1)),dtype=torch.float64)
    r=1.23*g+torch.tensor(rng.normal(0,.03,(200,1)))
    w=torch.tensor(rng.uniform(.1,3,(200,1))) if weighted else torch.ones_like(g)
    q=profile_Q(r,g,w).item()
    expected=(w*g*r).sum()/(w*g*g).sum()
    result=minimize_scalar(lambda x:float((w*(r-x*g)**2).sum()),bracket=(.8,1.4),method="brent",options={"xtol":1e-14})
    assert q==pytest.approx(expected.item(),abs=2e-15)
    assert q==pytest.approx(result.x,abs=2e-8)
    numeric=least_squares(lambda p:(w.sqrt()*(r-p[0]*g)).numpy().ravel(),[.7],
        jac=lambda p:-(w.sqrt()*g).numpy().reshape(-1,1),bounds=(0,np.inf),
        ftol=1e-14,xtol=1e-14,gtol=1e-14)
    assert q==pytest.approx(numeric.x[0],abs=2e-14)
    assert abs(float((w*g*(r-q*g)).sum()))<1e-12


def test_projection_envelope_gradient_matches_detached_optimum():
    theta=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    z=torch.linspace(.1,1,30,dtype=torch.float64)
    r=theta*z**2+.2;g=torch.exp(-theta*z)
    q=profile_Q(r,g)
    differentiable=torch.autograd.grad(((r-q*g)**2).mean(),theta,retain_graph=True)[0]
    detached=torch.autograd.grad(((r-q.detach()*g)**2).mean(),theta)[0]
    torch.testing.assert_close(differentiable,detached,rtol=1e-13,atol=1e-13)


def test_profile_positive_boundary_and_degeneracy():
    g=torch.ones(10,dtype=torch.float64)
    assert profile_Q(-g,g)==torch.finfo(g.dtype).tiny
    with pytest.raises(ValueError,match="energy"):
        profile_Q(g,g*0)
    with pytest.raises(ValueError,match="Weights"):
        profile_Q(g,g,g*0)
    with pytest.raises(FloatingPointError):
        profile_Q(g*float("nan"),g)


def test_profile_no_truth_and_Q_not_parameter():
    assert list(inspect.signature(profile_Q).parameters)==["R0","G","weights"]
    source=ProfiledGaussianSource(.5,.5,1.1,.08)
    assert set(dict(source.named_parameters()))=={"raw_x","raw_y"}
    assert "profiled_Q" in dict(source.named_buffers())
    with pytest.raises(TypeError):
        profile_Q(torch.ones(3),torch.ones(3),truth=1.)


def test_readonly_instrumentation_preserves_frozen_J0_bitwise():
    data=sparse();settings,formulation=recipe()
    plain=fit_candidate(data,settings,.08,42,formulation)
    audit=GradientPathway(data,1)
    instrumented=fit_candidate(data,settings,.08,42,formulation,on_objective=audit)
    for a,b in zip(plain[:2],instrumented[:2]):
        for key,value in a.state_dict().items():
            assert torch.equal(value,b.state_dict()[key])
    assert plain[3]==instrumented[3]
    assert all(r["data_source_gradient_norm"]==0 for r in audit.rows)


@pytest.mark.parametrize("block",["field","source"])
def test_actual_optimizer_step_freezes_other_block(block):
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.physics.autodiff import autograd_pde_residual
    torch.manual_seed(42)
    model=ConstrainedConcentrationMLP(.8,8,2,"hard")
    source=ProfiledGaussianSource(.5,.5,1.1,.08)
    activate_block(model,source,block)
    frozen=source if block=="field" else model
    before={k:v.clone() for k,v in frozen.state_dict().items()}
    active=model if block=="field" else source
    active_before=[p.detach().clone() for p in active.parameters()]
    optimizer=torch.optim.Adam([p for p in active.parameters() if p.requires_grad],lr=.001)
    points=torch.rand((20,3)).requires_grad_()
    r0=autograd_pde_residual(model(points),points,u=.35,v=-.15,D=.005,S=0.)
    if block=="source":
        r0=r0.detach()
    g=source.unit(points[:,:1],points[:,1:2])
    q=source.Q if block=="field" else profile_Q(r0,g)
    ((r0-q*g)**2).mean().backward();optimizer.step()
    assert all(torch.equal(v,frozen.state_dict()[k]) for k,v in before.items())
    assert any(not torch.equal(a,b) for a,b in zip(active_before,active.parameters()))


@pytest.mark.parametrize("algorithm",["differentiable_variable_projection","alternating"])
def test_candidates_deterministic_and_exact_constraints(algorithm):
    settings,formulation=recipe();data=sparse()
    candidate=dict(algorithm=algorithm,total_updates=4,cycles=2,field_updates=1,source_updates=1)
    profiling=dict(terminal_fitting_points=32,terminal_generator_seed_offset=2)
    results=[fit_profiled(data,settings,.08,42,formulation,candidate,profiling) for _ in range(2)]
    for a,b in zip(results[0][:2],results[1][:2]):
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    model,source=results[0][:2]
    p=torch.rand((20,3));assert (model(p)>=0).all();assert source.Q>0
    for axis,value in ((0,0.),(0,1.),(1,0.),(1,1.),(2,0.)):
        edge=p.clone();edge[:,axis]=value
        assert torch.count_nonzero(model(edge))==0


def test_J3_uses_only_observations_and_location(monkeypatch):
    import inverpinn.training.hybrid_source as hybrid
    data=sparse();h=np.arange(1,7)/10
    data=SparseTrainingData(data.positions,data.times,torch.tensor((1.25*h).reshape(3,2)),data.u,data.v,data.D)
    monkeypatch.setattr(hybrid,"forward",lambda *args:h)
    estimate,check=observation_profile(data,{"x_s":.3,"y_s":.7},{},.08)
    assert estimate["Q"]==pytest.approx(1.25,abs=1e-14)
    assert abs(check["analytic_numerical_difference"])<1e-10
    with pytest.raises(TypeError):
        observation_profile(data,{"x_s":.3,"y_s":.7,"Q_true":1.25},{},.08)


def test_new_plan_is_unique_reproducible_LHS_and_not_consumed():
    config=yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    first,second=development_plan(config),development_plan(config)
    assert first==second and len(first)==24
    old,_=previous_plans()
    for key in ("source_hash","generation_seed","scenario_id"):
        assert not {r[key] for r in first}&{r[key] for r in old}
    for key,lo,hi in (("x_s",.25,.75),("y_s",.25,.75),("Q",.8,1.4)):
        assert sorted(int((r["source"][key]-lo)/(hi-lo)*24) for r in first)==list(range(24))


def test_consumed_paths_truth_and_aliases_blocked(tmp_path):
    root=ROOT/"results/inverse_formulation_revision"
    with pytest.raises(ValueError):
        case_path(root,"fresh_001")
    with pytest.raises(ValueError):
        case_path(root,IDS[0],"../../../fresh_blind_benchmark")
    with pytest.raises(PermissionError):
        check_open(root,ROOT/"results/observability_diagnosis/pinn_results.csv")
    with pytest.raises(PermissionError):
        check_open(root,case_path(root,IDS[0],"truth/source.json"),IDS[0])
    with pytest.raises(PermissionError):
        check_open(root,root/"runs/J0"/IDS[0]/"seed_42/metrics.json",IDS[0])
    check_open(root,case_path(root,IDS[0],"inputs/observations.npz"),IDS[0])


def test_frozen_original_and_recovery_unchanged():
    assert sha256(ROOT/"configs/B_revised.yaml")==REVISED_HASH
    config=yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    assert config["optimization"]["primary_seed"]==42
    assert config["recovery"]["secondary_localization_max"]==.08
    assert config["recovery"]["secondary_relative_strength_max"]==.20
    assert config["candidates"]["J2"]["total_updates"]==12000


def test_actual_alternating_trainer_freezes_each_block():
    settings,formulation=recipe();data=sparse()
    candidate=dict(algorithm="alternating",total_updates=4,cycles=2,field_updates=1,source_updates=1)
    captured={};checked=[]
    def before(step,losses,weights,model,source,interior):
        captured.update(model=model,source=source,field=[p.detach().clone() for p in model.parameters()],
            location=[p.detach().clone() for p in source.parameters()])
    def after(row):
        if row["block"]=="field":
            assert all(torch.equal(a,b) for a,b in zip(captured["location"],captured["source"].parameters()))
        else:
            assert all(torch.equal(a,b) for a,b in zip(captured["field"],captured["model"].parameters()))
        checked.append(row["block"])
    fit_profiled(data,settings,.08,42,formulation,candidate,dict(terminal_fitting_points=32,terminal_generator_seed_offset=2),on_step=after,on_objective=before)
    assert checked==["field","source","field","source"]


def test_hybrid_field_preserves_IC_BC_and_labels_numerical_residual():
    from inverpinn.training.hybrid_source import numerical_field
    data=sparse();solver=dict(nx=21,ny=21,dt=.001,steps=800)
    fields,x,y,diag=numerical_field(data,dict(x_s=.3,y_s=.7,Q=1.),solver,.08,[.1,.2,.4,.8])
    assert fields.shape==(4,21,21) and fields.min()>=0
    assert diag["ic_rmse"]==diag["bc_rmse"]==0
    assert np.isfinite(diag["pde_rmse"])
    assert diag["pde_diagnostic_method"]=="native_centered_FD_not_neural_autograd"


def test_no_silent_overwrite_or_manifest_mutation(tmp_path,monkeypatch):
    import inverpinn.data.development_v3 as dev
    with pytest.raises(FileExistsError):
        dev.freeze({},tmp_path)
    # Exercise the production manifest checker on one tiny synthetic fixture.
    source=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08)
    plan=[dict(scenario_id=IDS[0],source=source)]
    monkeypatch.setattr(dev,"verify",lambda root:({},plan,{}))
    case=tmp_path/"datasets"/IDS[0];(case/"inputs").mkdir(parents=True);(case/"truth").mkdir()
    for name in ("inputs/observations.npz","inputs/physics.json","truth/checkpoint.npz"):
        (case/name).write_bytes(b"fixture")
    row=dict(scenario_id=IDS[0],observations_hash=sha256(case/"inputs/observations.npz"),
        physics_hash=sha256(case/"inputs/physics.json"),reference_data_hash=sha256(case/"truth/checkpoint.npz"),
        reference_path=f"datasets/{IDS[0]}/truth/checkpoint.npz",max_reference_error_estimate=.001)
    (case/"generation.json").write_text(json.dumps(row))
    assert dev.manifest(tmp_path)==dev.manifest(tmp_path)
    (tmp_path/"development_manifest.csv").write_text("altered")
    with pytest.raises(ValueError,match="Immutable"):
        dev.manifest(tmp_path)


def test_preservation_checker_detects_changed_old_bytes(tmp_path,monkeypatch):
    import inverpinn.data.development_v3 as dev
    (tmp_path/"consumed_hashes.json").write_text(json.dumps({"old":"abc"}))
    monkeypatch.setattr(dev,"inventory",lambda root:{"old":"abc"})
    assert dev.verify_consumed(tmp_path)==1
    monkeypatch.setattr(dev,"inventory",lambda root:{"old":"changed"})
    with pytest.raises(ValueError,match="Historical"):
        dev.verify_consumed(tmp_path)


def test_selection_rejects_single_median_improvement():
    import pandas as pd
    from inverpinn.evaluation.inverse_formulation import candidate_statistics,selection_checks
    config=yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    rows=[dict(localization_error=.01,relative_strength_error=.08,signed_Q_error=-.08,relative_l2=.1,rmse=.01,mae=.005,
        pde_rmse=.005,runtime_seconds=1,computational_success=True,evaluation_success=True,recovery_success=True,
        negative_fraction=0.,minimum_concentration=0.,bc_maxabs=0.,ic_maxabs=0.) for _ in range(24)]
    baseline=candidate_statistics(pd.DataFrame(rows),1e-6)
    frame=pd.DataFrame(rows);frame["relative_strength_error"]=.01;frame["signed_Q_error"]=[-.01,.01]*12
    frame.loc[0,"localization_error"]=.8
    stats=candidate_statistics(frame,1e-6)
    checks=selection_checks(stats,baseline,stats,baseline,config["selection"])
    assert checks["median_Q_error"] and not checks["localization_error_worst_noninferior"]


def test_end_to_end_short_candidates_and_failed_run_retention(tmp_path,monkeypatch):
    """Tiny integration fixture, never a scientific scenario or tuning result."""
    import sys
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_inverse_formulation_revision as runner
    from dataclasses import asdict
    from inverpinn.physics.finite_difference import solve_advection_diffusion
    from inverpinn.data.single_source_benchmark import object_hash
    config=yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    config["inference_solver"]=dict(nx=21,ny=21,dt=.001,steps=800)
    config["diagnostics"].update(interior_count=16,initial_count=8,boundary_per_edge=4)
    config["profiling"]["terminal_fitting_points"]=32
    config["candidates"]["J1"]["total_updates"]=4
    config["candidates"]["J2"].update(total_updates=4,cycles=2,field_updates=1,source_updates=1)
    config["gradient_interval"]=1;config["plot_dpi"]=40
    (tmp_path/"protocol.yaml").write_text(yaml.safe_dump(config))
    (tmp_path/"observability.csv").write_text("fixture")
    (tmp_path/"information_complete.json").write_text(json.dumps(dict(observability_hash=sha256(tmp_path/"observability.csv"))))
    setting,form=recipe()
    training=setting.optimizer_config(42);training.pop("experiment")
    training.update(initial_x=.5,initial_y=.5,initial_Q=1.1)
    monkeypatch.setattr(runner,"recipe_for",lambda *args:dict(training=training,formulation=form,sigma=.08,seed=42))
    monkeypatch.setattr(runner,"install_guard",lambda root:{"fitting_case":None})
    identifier=IDS[0];case=tmp_path/"datasets"/identifier
    (case/"inputs").mkdir(parents=True);(case/"truth").mkdir()
    truth=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08)
    positions=torch.tensor([[.2,.4],[.6,.7]],dtype=torch.float64)
    solution=solve_advection_diffusion(**config["inference_solver"],**config["physics"],sources=[truth],
        output_mode="selected_times",output_times=config["evaluation_times"],sensor_positions=positions,measurement_times=[0.,.4,.8])
    np.savez_compressed(case/"inputs/observations.npz",positions=positions.numpy(),times=np.array([0.,.4,.8]),measurements=solution["measurements"].numpy())
    (case/"inputs/physics.json").write_text(json.dumps(dict(transport=config["physics"],domain=config["domain"],
        initial_condition="C=0 everywhere",boundary_condition="C=0 on all four edges",units={})))
    (case/"truth/source.json").write_text(json.dumps(truth))
    np.savez_compressed(case/"truth/reference.npz",**{k:v.numpy() for k,v in solution.items()})
    hashes=dict(scenario_id=identifier,observations_hash=sha256(case/"inputs/observations.npz"),physics_hash=sha256(case/"inputs/physics.json"))
    (tmp_path/"input_manifest.json").write_text(json.dumps([hashes]))
    (case/"generation.json").write_text(json.dumps(hashes|dict(reference_path=f"datasets/{identifier}/truth/reference.npz",
        reference_data_hash=sha256(case/"truth/reference.npz"),reference_target_met=True,max_reference_error_estimate=.001)))
    for candidate in ("J0","J1","J2","J3"):
        runner.fit_one(tmp_path,candidate,identifier,42)
        run=tmp_path/"runs"/candidate/identifier/"seed_42"
        row=json.loads((run/"metrics.json").read_text())
        assert row["evaluation_success"] and row["observations_hash"]==hashes["observations_hash"]
        assert row["bc_maxabs"]==row["ic_maxabs"]==row["negative_fraction"]==0
        assert "preserved" in runner.fit_one(tmp_path,candidate,identifier,42)
    def fail(*args,**kwargs):
        raise FloatingPointError("deliberate test failure")
    monkeypatch.setattr(runner,"fit_candidate",fail)
    runner.fit_one(tmp_path,"J0",identifier,142)
    failed=json.loads((tmp_path/"runs/J0"/identifier/"seed_142/metrics.json").read_text())
    assert not failed["computational_success"] and not failed["recovery_success"]
    assert "deliberate test failure" in failed["failure"]
    assert (tmp_path/"runs/J0"/identifier/"seed_42/metrics.json").exists()


def test_report_pairs_observations_and_keeps_failed_rows(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import report_inverse_formulation_revision as reporter
    config=yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    monkeypatch.setattr(reporter,"verify",lambda root:(config,[],{"protocol_hash":"fixture"}))
    monkeypatch.setattr(reporter,"manifest",lambda root:None)
    monkeypatch.setattr(reporter,"verify_consumed",lambda root:9031)
    monkeypatch.setattr(reporter,"figures",lambda *args:None)
    monkeypatch.setattr(reporter,"snapshot",lambda *args:None)
    monkeypatch.setattr(reporter,"oracle_location_diagnostic",lambda *args:{"label":"fixture"})
    (tmp_path/"B_revised_frozen.yaml").write_text("fixture")
    (tmp_path/"fitting_execution").mkdir()
    (tmp_path/"fitting_execution/source_snapshot.zip").write_bytes(b"fixture")
    (tmp_path/"observability.csv").write_text("scenario_id,information_group\n"+"\n".join(f"{k},{'lower' if i<12 else 'higher'}" for i,k in enumerate(IDS)))
    repeats=[]
    for name in ("J0","J1","J2","J3"):
        rows=[]
        for i,identifier in enumerate(IDS):
            row=dict(candidate=name,scenario_id=identifier,optimization_seed=42,primary=True,
                computational_success=True,evaluation_success=True,recovery_success=False,observations_hash=f"same_{i}",
                x_true=.3,y_true=.7,Q_true=1.1,x_pred=.4,y_pred=.6,Q_pred=.8,signed_Q_error=-.3,
                localization_error=.14,relative_strength_error=.27,relative_l2=.3,rmse=.01,mae=.005,
                pde_rmse=.005,runtime_seconds=1.,negative_fraction=0.,minimum_concentration=0.,bc_maxabs=0.,ic_maxabs=0.,
                reference_target_met=True,reference_error_estimate=.003,conditional_Q_at_estimated_location=1.,
                PDE_Q_at_fixed_field=.81,field_observation_Q_gap=-.19)
            if name=="J1" and i==23:
                row.update(computational_success=False,evaluation_success=False)
            rows.append(row)
            if identifier in config["optimization"]["sensitivity_ids"]:
                repeats.extend([row|dict(optimization_seed=seed,primary=seed==42) for seed in (42,142,242)])
            if name!="J3":
                run=tmp_path/"runs"/name/identifier/"seed_42";run.mkdir(parents=True)
                pd.DataFrame([dict(epoch=epoch,pre_Q=q,full_sensor_mse=loss,data_pde_gradient_cosine=-.5,
                    raw_x_gradient_norm=.1,raw_y_gradient_norm=.1,data_source_gradient_norm=0.)
                    for epoch,q,loss in ((1,.9,.2),(100,.8,.1))]).to_csv(run/"gradient_pathway.csv",index=False)
                (run/"training_history.jsonl").write_text(json.dumps({"effective_Q_update":-.1,"source/Q":.8})+"\n")
        pd.DataFrame(rows).to_csv(tmp_path/f"{name}_results.csv",index=False)
    pd.DataFrame(repeats).to_csv(tmp_path/"seed_sensitivity.csv",index=False)
    (tmp_path/"integrity_audit.json").write_text(json.dumps(dict(audited_runs=128,model_and_selection_code_unchanged=True,
        table_hashes={p.name:sha256(p) for p in [*(tmp_path/f"{n}_results.csv" for n in ("J0","J1","J2","J3")),tmp_path/"seed_sensitivity.csv"]})))
    reporter.report(tmp_path)
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["selected_candidate"] is None
    assert summary["candidates"]["J1"]["failed_fits"]==1
    assert summary["candidates"]["J1"]["count"]==24
    assert not summary["selection_checks"]["J1"]["fits_complete"]
    association=summary["exploratory_J3_location_error_spearman"]["relative_strength_error"]
    assert association["rho"] is None and association["n"]==24
    assert association["status"]=="undefined_insufficient_pairs_or_constant_ranks"
    assert len(pd.read_csv(tmp_path/"paired_results.csv"))==72
    with pytest.raises(FileExistsError):
        reporter.report(tmp_path)


def test_postfit_audit_rejects_independent_truth_mutation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    from audit_inverse_formulation_revision import verify_truth_against_plan,checked_difference
    from inverpinn.data.single_source_benchmark import object_hash
    truth=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08)
    generation=dict(source_hash=object_hash(truth))
    verify_truth_against_plan(truth,generation,truth)
    with pytest.raises(ValueError,match="Source truth"):
        verify_truth_against_plan(truth|dict(Q=1.1),generation,truth)
    with pytest.raises(ValueError,match="Source truth"):
        verify_truth_against_plan(truth,generation,truth|dict(x_s=.4))
    with pytest.raises(ValueError,match="Recomputed"):
        checked_difference(.1,.2,"fixture metric")
