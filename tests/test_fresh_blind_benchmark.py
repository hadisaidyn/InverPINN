"""Prospective independence, frozen-model and confirmatory-analysis contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inverpinn.data import fresh_blind as fresh
from inverpinn.data.single_source_benchmark import object_hash, scenario_plan
from inverpinn.data.synthetic_reference import ROOT, sha256


@pytest.fixture
def config():
    return yaml.safe_load((ROOT/"configs/fresh_blind_benchmark.yaml").read_text())


@pytest.fixture
def legacy():
    return yaml.safe_load((ROOT/"configs/single_source_benchmark.yaml").read_text())


def test_all_thirty_sources_ids_seeds_are_new_and_reproducible(config, legacy):
    a, b = fresh.fresh_plan(config, legacy), fresh.fresh_plan(config, legacy)
    old = scenario_plan(legacy)
    assert a == b and len(a) == 30
    for key in ("scenario_id", "source_hash", "generation_seed"):
        values = {r[key] for r in a}
        assert len(values) == 30 and not values & {r[key] for r in old}
    for row in a:
        assert .25 <= row["source"]["x_s"] <= .75
        assert .25 <= row["source"]["y_s"] <= .75
        assert .8 <= row["source"]["Q"] <= 1.4


@pytest.mark.parametrize("old_split", ["development", "heldout"])
def test_old_master_seed_rejected_without_resampling(config, legacy, old_split):
    config["scenario_sampling"]["seed"] = legacy["scenario_sampling"][old_split]["seed"]
    with pytest.raises(ValueError, match="new master"):
        fresh.fresh_plan(config, legacy)


def test_model_configs_and_original_code_match_frozen_commit(config):
    assert sha256(ROOT/config["models"]["B_revised"]["config"]) == fresh.REVISED_HASH
    assert sha256(ROOT/config["models"]["B0"]["config"]) == fresh.B0_HASH
    assert len(fresh.original_hashes(config)) > 10
    config["models"]["B_revised"]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="configuration changed"):
        fresh.original_hashes(config)


@pytest.mark.parametrize("identifier", ["development_001", "heldout_001", "heldout_030", "fresh_031", "../fresh_001"])
def test_old_ids_cannot_enter_fresh_dataset_path(tmp_path, identifier):
    with pytest.raises(ValueError, match="Only fresh"):
        fresh.fresh_path(tmp_path, identifier)


def test_no_escape_or_symlink_to_old_case(tmp_path):
    (tmp_path/"datasets").mkdir()
    other = tmp_path/"old"
    other.mkdir()
    (tmp_path/"datasets/fresh_001").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        fresh.fresh_path(tmp_path, "fresh_001")
    with pytest.raises(ValueError, match="escapes"):
        fresh.fresh_path(tmp_path, "fresh_002", "../../old")


def test_old_result_and_fitting_truth_io_guard():
    root = ROOT/"results/fresh_blind_benchmark"
    for path in (ROOT/"results/single_source_benchmark/summary.json", ROOT/"results/model_diagnosis/raw_results.csv"):
        with pytest.raises(PermissionError, match="Previous"):
            fresh.check_open(root, path)
    for name in ("scenario_plan.json", "scenario_manifest.csv", "datasets/fresh_001/truth/source.json",
                 "datasets/fresh_002/inputs/observations.npz"):
        with pytest.raises(PermissionError):
            fresh.check_open(root, root/name, "fresh_001")
    fresh.check_open(root, root/"datasets/fresh_001/inputs/observations.npz", "fresh_001")


def test_primary_and_seed_substudy_are_disjoint_and_fixed(config):
    jobs = fresh.jobs(config)
    assert len(jobs) == 80
    assert all(seed == 42 for _, _, seed in jobs[:60])
    assert {m for m, _, _ in jobs[:60]} == {"B0", "B_revised"}
    for identifier in config["optimization"]["additional_seed_scenarios"]:
        assert {s for m, case, s in jobs if m == "B_revised" and case == identifier} == {42, 314, 2718}
    assert len({tuple(j) for j in jobs}) == 80


def test_unchanged_recovery_threshold(config, legacy):
    assert config["recovery"]["secondary_localization_max"] == legacy["recovery"]["secondary_localization_max"] == .08
    assert config["recovery"]["secondary_relative_strength_max"] == legacy["recovery"]["secondary_relative_strength_max"] == .20


def test_manifest_freezes_all_cases_with_source_truth_only_in_separate_table(tmp_path, monkeypatch, config, legacy):
    plan = fresh.fresh_plan(config, legacy)
    monkeypatch.setattr(fresh, "load_frozen", lambda _: (config, plan, {"protocol_hash":"fixture"}))
    for row in plan:
        case = tmp_path/"datasets"/row["scenario_id"]
        case.mkdir(parents=True)
        (case/"generation.json").write_text(json.dumps(dict(scenario_id=row["scenario_id"], observations_hash="obs", physics_hash="physics")))
    rows = fresh.manifest(tmp_path)
    assert len(rows) == 30 and "Q_true" in rows[0]
    inputs = json.loads((tmp_path/"input_manifest.json").read_text())
    assert set(inputs[0]) == {"scenario_id", "observations_hash", "physics_hash"}
    assert fresh.manifest(tmp_path) == rows
    with (tmp_path/"scenario_manifest.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="Immutable"):
        fresh.manifest(tmp_path)


def test_existing_benchmark_never_overwritten(tmp_path, config):
    with pytest.raises(FileExistsError, match="immutable"):
        fresh.freeze(config, tmp_path)


def fake_rows(config):
    """Unrelated small fake metric rows; never load research outcomes in tests."""
    import pandas as pd
    from inverpinn.evaluation.fresh_blind import METRICS
    rows=[]
    for model,identifier,seed in fresh.jobs(config):
        row={key:.1 for key in METRICS}
        row.update(model=model,scenario_id=identifier,optimization_seed=seed,computational_success=True,
                   evaluation_success=True,recovery_success=True,reference_target_met=True,
                   observations_hash=f"obs-{identifier}",reference_data_hash=f"ref-{identifier}",
                   x_true=.5,y_true=.5,Q_true=1.,Q_pred=.9)
        rows.append(row)
    return pd.DataFrame(rows)


def test_additional_seeds_cannot_replace_bad_primary_and_failures_remain(config):
    from inverpinn.evaluation.fresh_blind import primary_rows, summarize_model
    rows=fake_rows(config)
    primary=(rows.model=="B_revised") & (rows.scenario_id=="fresh_001") & (rows.optimization_seed==42)
    rows.loc[primary,["localization_error","relative_strength_error","relative_l2"]]=np.nan
    rows.loc[primary,["computational_success","evaluation_success","recovery_success"]]=False
    subset=primary_rows(rows,config,"B_revised")
    result=summarize_model(subset,config)
    assert len(subset)==30 and result["recovered"]==29 and result["recovery_rate"]==29/30
    assert result["metrics"]["localization_error"]["missing"]==1
    assert np.isnan(subset.iloc[0].localization_error)


def test_missing_or_duplicated_primary_rows_rejected(config):
    import pandas as pd
    from inverpinn.evaluation.fresh_blind import primary_rows
    rows=fake_rows(config)
    with pytest.raises(ValueError,match="30 unique"):
        primary_rows(rows.iloc[1:],config,"B0")
    with pytest.raises(ValueError,match="30 unique"):
        primary_rows(pd.concat([rows,rows.iloc[[0]]]),config,"B0")


def test_paired_comparison_matches_ids_observations_and_preserves_missing(config):
    from inverpinn.evaluation.fresh_blind import paired_analysis, primary_rows
    rows=fake_rows(config)
    a,b=(primary_rows(rows,config,m) for m in ("B0","B_revised"))
    b.loc[0,"localization_error"] = .01
    b.loc[1,"localization_error"] = np.nan
    table,summary=paired_analysis(a,b.iloc[::-1],config["analysis"])
    assert table.iloc[0].delta_localization_error==pytest.approx(-.09)
    assert summary["localization_error"]["B_revised_better"]==1
    assert summary["localization_error"]["missing"]==1
    assert summary["localization_error"]["tied"]==28
    b.loc[0,"observations_hash"]="different-observations"
    with pytest.raises(ValueError,match="observations_hash"):
        paired_analysis(a,b,config["analysis"])


def test_preregistered_ties_missing_and_direction(config):
    from inverpinn.evaluation.fresh_blind import compare_values
    settings=config["analysis"]
    args=(settings["metric_tie_atol"],settings["metric_tie_rtol"])
    assert compare_values(1.,1.+1e-7,*args)[1]=="tied"
    assert compare_values(1.,1.01,*args)[1]=="B0_better"
    assert compare_values(1.,.99,*args)[1]=="B_revised_better"
    assert compare_values(np.nan,.99,*args)==(None,"missing")


def test_paired_bootstrap_is_reproducible_and_constant_difference_exact():
    from inverpinn.evaluation.fresh_blind import bootstrap_difference
    result=bootstrap_difference([-.3]*30,7,1000)
    np.testing.assert_allclose(result["mean_CI95"],[-.3,-.3],atol=1e-14)
    assert bootstrap_difference([1,2,5],7,1000)==bootstrap_difference([1,2,5],7,1000)


def test_seed_sensitivity_preserves_all_three_and_does_not_select(config):
    from inverpinn.evaluation.fresh_blind import seed_sensitivity
    rows=fake_rows(config)
    rows.loc[(rows.scenario_id=="fresh_001") & (rows.model=="B_revised") & (rows.optimization_seed==314),"Q_pred"]=.3
    table,result=seed_sensitivity(rows,config)
    assert len(table)==30 and len(result)==10
    assert result[0]["Q_pred_range"]==pytest.approx(.6)
    assert not any("best" in k or "selected" in k for k in result[0])


def test_Q_bias_tolerance_and_signed_distribution(config):
    from inverpinn.evaluation.fresh_blind import primary_rows,summarize_model
    rows=primary_rows(fake_rows(config),config,"B_revised")
    rows.loc[0,"Q_pred"]=1.+1e-8
    rows.loc[1,"Q_pred"]=1.2
    bias=summarize_model(rows,config)["Q_bias"]
    assert (bias["under"],bias["over"],bias["approximately_equal"])==(28,1,1)
    assert bias["signed_error"]["median"]==pytest.approx(-.1)


def test_profiled_Q_is_analytic_optimum_uses_no_truth_and_respects_nonnegative_domain():
    from inverpinn.evaluation.fresh_blind import profile_strength
    g=np.array([[1.,2.],[3.,4.]])
    q,mse=profile_strength(g,1.23*g)
    assert q==pytest.approx(1.23) and mse==pytest.approx(0.,abs=1e-25)
    for candidate in [0.,.5,1.,1.5,3.]:
        assert mse <= np.mean((candidate*g-1.23*g)**2)
    assert profile_strength(g,-g)[0]==0
    with pytest.raises(ValueError,match="Zero source"):
        profile_strength(g*0,g)


def test_profile_respects_linearity_of_unchanged_forward_operator():
    from inverpinn.physics.finite_difference import solve_advection_diffusion
    from inverpinn.evaluation.fresh_blind import profile_strength
    import torch
    settings=dict(nx=11,ny=11,dt=.002,steps=10,u=.35,v=-.15,D=.005,
        output_mode="sensor_only",sensor_positions=torch.tensor([[.3,.7],[.5,.5]],dtype=torch.float64),
        measurement_times=[0.,.01,.02])
    source=dict(x_s=.4,y_s=.6,Q=1.,sigma=.08)
    g=solve_advection_diffusion(**settings,sources=[source])["measurements"].numpy()
    y=solve_advection_diffusion(**settings,sources=[source|dict(Q=1.3)])["measurements"].numpy()
    np.testing.assert_allclose(y,1.3*g,rtol=1e-13,atol=1e-15)
    q,error=profile_strength(g,y)
    assert q==pytest.approx(1.3,rel=1e-12) and error < 1e-25


def test_interpretation_cannot_hide_high_failure_rate_or_physical_invalidity(config):
    from inverpinn.evaluation.fresh_blind import primary_rows,summarize_model,interpretation
    rows=fake_rows(config)
    a=primary_rows(rows,config,"B0")
    b=primary_rows(rows,config,"B_revised")
    b.loc[:,["negative_fraction","bc_maxabs","ic_maxabs"]]=0.
    b.loc[:,["localization_error","relative_strength_error","relative_l2"]]=.05
    base=summarize_model(a,config)
    revised=summarize_model(b,config)
    assert interpretation(base,revised,config)=="strong controlled recovery"
    revised["recovered"]=20
    assert interpretation(base,revised,config)=="conditional controlled recovery"
    revised["recovered"]=17
    assert interpretation(base,revised,config)=="unreliable controlled recovery"
    revised["recovered"]=30
    revised["metrics"]["negative_fraction"]["max"]=.001
    assert interpretation(base,revised,config)=="unreliable controlled recovery"


@pytest.fixture
def fit_runner(monkeypatch):
    import importlib.util
    scripts=ROOT/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("fresh_fit_runner",scripts/"run_fresh_blind_benchmark.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_uses_exact_frozen_training_and_only_requested_seed(tmp_path,config,fit_runner):
    (tmp_path/"protocol.yaml").write_text(yaml.safe_dump(config))
    for name in ("B0","B_revised"):
        (tmp_path/f"{name}_frozen.yaml").write_bytes((ROOT/config["models"][name]["config"]).read_bytes())
    a=fit_runner.recipe_for(tmp_path,"B0",42)
    b=fit_runner.recipe_for(tmp_path,"B_revised",42)
    c=fit_runner.recipe_for(tmp_path,"B_revised",314)
    assert a["training"]["steps"]==6000 and b["training"]["steps"]==12000
    assert b["training"]["initial_Q"]==1.1 and a["training"]["initial_Q"]==.5
    assert c==b|dict(seed=314)
    assert set(b)=={"training","formulation","sigma","seed"}
    assert fit_runner.run_path(tmp_path,"B_revised","fresh_001",42)!=fit_runner.run_path(tmp_path,"B_revised","fresh_001",314)


def test_evaluation_refuses_truth_before_terminal_attempt(tmp_path,config,fit_runner):
    run=fit_runner.run_path(tmp_path,"B_revised","fresh_001",42)
    run.mkdir(parents=True)
    (run/"attempt.json").write_text(json.dumps(dict(training_terminated=False)))
    with pytest.raises(RuntimeError,match="terminal"):
        fit_runner.evaluate(tmp_path,"B_revised","fresh_001",42,config)


def test_fitting_interfaces_have_no_truth_metadata_or_path_parameters():
    import inspect
    from inverpinn.training.single_source import fit_sparse_source,SparseTrainingData
    from inverpinn.training.development_candidates import fit_candidate
    for fn in (fit_sparse_source,fit_candidate):
        assert not set(inspect.signature(fn).parameters)&{"truth","scenario","reference","source_truth","manifest"}
    assert set(SparseTrainingData.__dataclass_fields__)=={"positions","times","measurements","u","v","D"}


def test_required_figures_render_with_fake_all_case_tables(tmp_path,config):
    import pandas as pd
    from inverpinn.evaluation.fresh_blind import primary_rows,paired_analysis,seed_sensitivity
    from inverpinn.visualization.fresh_blind import figures
    config["plot_dpi"]=40  # rendering smoke test only, never the benchmark config
    rows=fake_rows(config)
    rows["x_pred"],rows["y_pred"]=.49,.51
    a,b=(primary_rows(rows,config,m) for m in ("B0","B_revised"))
    paired,_=paired_analysis(a,b,config["analysis"])
    seeds,_=seed_sensitivity(rows,config)
    (tmp_path/"sensor_geometry.json").write_text(json.dumps([[.2,.3],[.6,.7]]))
    landscape=tmp_path/"identifiability"
    landscape.mkdir()
    pd.DataFrame([dict(kind="grid",x=x,y=y,fixed_Q_mse=(x-.5)**2+(y-.5)**2+.001,
                       profiled_Q_mse=(x-.5)**2+(y-.5)**2+.0005) for y in [.25,.5,.75] for x in [.25,.5,.75]]).to_csv(landscape/"landscapes.csv",index=False)
    (landscape/"summary.json").write_text(json.dumps(dict(true_source=dict(x_s=.5,y_s=.5),primary_PINN_source=dict(x_s=.49,y_s=.51))))
    figures(tmp_path,a,b,paired,seeds,config)
    names=("fresh_true_vs_predicted_sources","fresh_localization_distribution","fresh_Q_true_vs_predicted",
           "fresh_concentration_error_distribution","fresh_B0_vs_Brevised_paired","fresh_physical_diagnostics",
           "fresh_seed_sensitivity","fresh_identifiability_fixed_Q","fresh_identifiability_profiled_Q")
    for name in names:
        assert (tmp_path/f"{name}.png").stat().st_size > 1000
        assert (tmp_path/f"{name}.pdf").stat().st_size > 1000


def test_prospective_Adam_defaults_match_installed_torch(config):
    import torch
    frozen=yaml.safe_load((ROOT/config["models"]["B_revised"]["config"]).read_text())["optimizer"]
    expected={"lr" if key=="learning_rate" else key:value for key,value in frozen.items() if key!="name"}
    expected["betas"]=tuple(expected["betas"])
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))],lr=frozen["learning_rate"])
    assert optimizer.defaults==expected


def test_live_process_guard_blocks_fitting_truth_and_previous_results(tmp_path):
    import subprocess,sys
    case=tmp_path/"datasets/fresh_001"
    (case/"truth").mkdir(parents=True)
    (case/"inputs").mkdir()
    (case/"truth/source.json").write_text('{"Q":1.0}')
    (case/"inputs/physics.json").write_text('{}')
    code='''
from pathlib import Path
from inverpinn.data.fresh_blind import install_guard
from inverpinn.data.synthetic_reference import ROOT
import sys
r=Path(sys.argv[1]); state=install_guard(r); state['fitting_case']='fresh_001'
assert (r/'datasets/fresh_001/inputs/physics.json').read_text() == '{}'
for path in [r/'datasets/fresh_001/truth/source.json',ROOT/'results/single_source_benchmark/summary.json']:
    try: path.read_text()
    except PermissionError: pass
    else: raise AssertionError('Forbidden read was permitted')
print('guard verified')
'''
    result=subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="guard verified"


@pytest.mark.parametrize("name",["B0","B_revised"])
def test_new_checkpoint_loader_preserves_frozen_model_and_source_exactly(tmp_path,config,fit_runner,name):
    import torch
    from inverpinn.models.mlp import ConcentrationMLP
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.physics.trainable_source import TrainableGaussianSource
    from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
    frozen=yaml.safe_load((ROOT/config["models"][name]["config"]).read_text())
    training=frozen["training"] if name=="B0" else frozen["inference"]["training"]
    form="baseline" if name=="B0" else frozen["inference"]["formulation"]
    torch.manual_seed(1234)
    if name=="B0":
        model=ConcentrationMLP([0.,0.,0.],[1.,1.,.8],training["width"],training["depth"])
        source=TrainableGaussianSource(.5,.5,.5,.08)
    else:
        model=ConstrainedConcentrationMLP(.8,training["width"],training["depth"],form["constraint"])
        source=DiagnosticGaussianSource(.5,.5,1.1,.08,form["Q_parameterization"])
    path=tmp_path/'model.pt'
    torch.save(dict(recipe=dict(training=training,formulation=form,sigma=.08,seed=1234),
                    model_state_dict=model.state_dict(),source_state_dict=source.state_dict()),path)
    copy_model,copy_source=fit_runner.load_fit(path)
    points=torch.rand(15,3)
    assert torch.equal(model(points),copy_model(points))
    assert torch.equal(source(points[:,:1],points[:,1:2]),copy_source(points[:,:1],points[:,1:2]))
