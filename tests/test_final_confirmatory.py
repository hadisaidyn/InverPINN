"""Frozen hypothesis, fresh inputs, immutable decisions and perturbation gates."""
import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from inverpinn.data.final_confirmatory import IDS,METHODS,ROOT,REVISION7,exact_J1,read_config,previous_sources,scenario_plan,check_open
from inverpinn.data.final_robustness import nested_sensors,noisy_observations,wind_conditions,require_confirmation
from inverpinn.evaluation.final_confirmatory import statistics,criteria,validate_rows,paired
from inverpinn.training.inverse_formulation import fit_profiled
from inverpinn.evaluation.observability import classical_inverse


def fixture_rows():
    return pd.DataFrame([dict(method=m,scenario_id=k,optimization_seed=42,observations_hash=k,reference_data_hash=k,
        computational_success=True,evaluation_success=True,recovery_success=True,localization_error=.01,relative_strength_error=.05,
        relative_l2=.08,rmse=.001,mae=.0005,pde_rmse=.005,pde_mae=.004,runtime_seconds=1.,signed_Q_error=.02,
        signed_relative_Q_error=.02,negative_fraction=0.,bc_maxabs=0.,ic_maxabs=0.) for m in METHODS for k in IDS])


def test_exact_revision7_recipe_and_code():
    import subprocess,hashlib
    assert read_config("J1_confirmatory.yaml")==exact_J1()
    for path in ("configs/B_revised.yaml","configs/inverse_formulation_revision.yaml","src/inverpinn/training/inverse_formulation.py",
                 "src/inverpinn/physics/profiled_source.py","src/inverpinn/models/constrained_concentration.py","src/inverpinn/models/mlp.py","src/inverpinn/evaluation/observability.py"):
        previous=subprocess.check_output(["git","show",f"{REVISION7}:{path}"],cwd=ROOT)
        assert hashlib.sha256(previous).digest()==hashlib.sha256((ROOT/path).read_bytes()).digest()
    j=exact_J1();assert j["recipe"]["seed"]==42 and j["candidate"]["total_updates"]==12000
    assert j["profiling"]["terminal_fitting_points"]==8192


def test_thirty_new_reproducible_scenarios():
    old,seeds,triples=previous_sources();assert len(old)==124
    config=read_config("final_confirmatory.yaml");plan=scenario_plan(config,old,seeds,triples)
    assert plan==scenario_plan(config,old,seeds,triples) and len(plan)==30
    for key in ("scenario_id","source_hash","generation_seed"):
        assert not {r[key] for r in plan}&{r[key] for r in old}
    for r in plan:assert .25<=r["source"]["x_s"]<=.75 and .25<=r["source"]["y_s"]<=.75 and .8<=r["source"]["Q"]<=1.4
    with pytest.raises(ValueError):scenario_plan(config,plan,seeds,triples)
    with pytest.raises(ValueError):scenario_plan(config,old,seeds|{config["scenario_sampling"]["seed"]},triples)


def test_truth_and_prior_outcomes_forbidden(tmp_path):
    for path in (tmp_path/"datasets"/IDS[0]/"truth/source.json",tmp_path/"runs/J1"/IDS[0]/"metrics.json",tmp_path/"scenario_plan.json"):
        with pytest.raises(PermissionError):check_open(tmp_path,path,IDS[0],"J1")
    with pytest.raises(PermissionError):check_open(tmp_path,ROOT/"results/inverse_formulation_revision/J1_results.csv")
    check_open(tmp_path,tmp_path/"datasets"/IDS[0]/"inputs/observations.npz",IDS[0],"J1")
    assert "truth" not in inspect.signature(fit_profiled).parameters
    assert "truth" not in inspect.signature(classical_inverse).parameters


def test_preregistered_gates_and_failed_rows_retained():
    config=read_config("final_confirmatory.yaml");rule=config["advancement"]
    assert (rule["min_recovered"],rule["median_localization_max"],rule["median_relative_Q_max"],rule["median_L2_max"],rule["PDE_median_max_ratio"])==(27,.02,.1,.15,1.1)
    frame=fixture_rows();validate_rows(frame)
    summarize=lambda: {m:statistics(frame[frame.method==m],config) for m in METHODS}
    assert all(criteria(summarize(),rule).values())
    frame.loc[(frame.method=="J1")&frame.scenario_id.isin(IDS[:4]),"recovery_success"]=False
    assert not criteria(summarize(),rule)["joint_recovery"]
    frame.loc[0,"computational_success"]=False
    assert not criteria(summarize(),rule)["all_runs_evaluable"]
    assert summarize()["J1"]["count"]==30
    with pytest.raises(ValueError):validate_rows(frame.iloc[1:])


@pytest.mark.parametrize("metric,value,gate",[("pde_rmse",.00551,"paired_median_PDE"),("signed_relative_Q_error",-.101,"signed_relative_Q_bias"),
    ("signed_Q_error",-.01,"nonuniversal_underestimation"),("negative_fraction",.001,"positivity"),("bc_maxabs",1e-12,"BC_exact"),("ic_maxabs",1e-12,"IC_exact")])
def test_no_rounding_or_relaxation(metric,value,gate):
    frame=fixture_rows();frame.loc[frame.method=="J1",metric]=value;config=read_config("final_confirmatory.yaml")
    assert not criteria({m:statistics(frame[frame.method==m],config) for m in METHODS},config["advancement"])[gate]


def test_pairing_seed_and_bootstrap():
    frame=fixture_rows();config=read_config("final_confirmatory.yaml")
    a,s=paired(frame,"B_revised",config["analysis"]);b,t=paired(frame,"B_revised",config["analysis"])
    assert s==t and s["relative_l2"]["tied"]==30 and len(a)==30
    frame.loc[0,"observations_hash"]="changed"
    with pytest.raises(ValueError):validate_rows(frame)
    frame=fixture_rows();frame.loc[0,"optimization_seed"]=142
    with pytest.raises(ValueError):validate_rows(frame)


def test_geometry_noise_and_wind_are_fixed():
    positions=np.array(json.loads((ROOT/"configs/single_source_sensors.json").read_text()))
    subsets=nested_sensors(positions)
    assert subsets==nested_sensors(positions) and subsets[5]==subsets[10][:5] and subsets[10]==subsets[20][:10]
    assert set(subsets[20])==set(range(20))
    clean=np.arange(60.).reshape(3,20)
    low=noisy_observations(clean,.05,2009202848,0);high=noisy_observations(clean,.2,2009202848,0)
    np.testing.assert_allclose(high-clean,4*(low-clean),atol=1e-14)
    np.testing.assert_array_equal(low,noisy_observations(clean,.05,2009202848,0))
    np.testing.assert_array_equal(clean,noisy_observations(clean,0.,2009202848,0))
    winds=wind_conditions();w=np.array([.35,-.15])
    for name,sign in (("direction_minus10",-1),("direction_plus10",1)):
        v=np.array(list(winds[name].values()));assert np.linalg.norm(v)==pytest.approx(np.linalg.norm(w))
        assert np.rad2deg(np.arctan2(v[1],v[0])-np.arctan2(w[1],w[0]))==pytest.approx(sign*10)
    np.testing.assert_allclose(list(winds["speed_minus10"].values()),w*.9)
    np.testing.assert_allclose(list(winds["speed_plus10"].values()),w*1.1)


def test_robustness_cannot_bypass_failed_confirmation(tmp_path,monkeypatch):
    import inverpinn.data.final_robustness as module
    monkeypatch.setattr(module,"verify",lambda root:({},[],{}))
    (tmp_path/"decision.json").write_text(json.dumps(dict(passed=False)))
    with pytest.raises(PermissionError,match="failed"):require_confirmation(tmp_path)


def test_manifest_is_locked_and_truth_cannot_change(tmp_path,monkeypatch):
    import inverpinn.data.final_confirmatory as module
    from inverpinn.data.single_source_benchmark import object_hash
    from inverpinn.data.synthetic_reference import sha256
    truth=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08);plan=[dict(scenario_id=IDS[0],source=truth,source_hash=object_hash(truth))]
    monkeypatch.setattr(module,"verify",lambda root:({},plan,{}))
    case=tmp_path/"datasets"/IDS[0];(case/"truth").mkdir(parents=True);(case/"inputs").mkdir()
    (case/"truth/source.json").write_text(json.dumps(truth))
    for name in ("inputs/observations.npz","inputs/physics.json","truth/reference.npz"):(case/name).write_bytes(b"fixture")
    record=dict(scenario_id=IDS[0],source_hash=object_hash(truth),observations_hash=sha256(case/"inputs/observations.npz"),physics_hash=sha256(case/"inputs/physics.json"),
        reference_data_hash=sha256(case/"truth/reference.npz"),reference_path=str((case/"truth/reference.npz").relative_to(tmp_path)),max_reference_error_estimate=.001)
    (case/"generation.json").write_text(json.dumps(record))
    assert module.manifest(tmp_path)==module.manifest(tmp_path)
    locked=(tmp_path/"manifest_lock.json").read_bytes()
    (case/"truth/source.json").write_text(json.dumps(truth|dict(Q=1.1)))
    with pytest.raises(ValueError,match="Truth"):module.manifest(tmp_path)
    (case/"truth/source.json").write_text(json.dumps(truth))
    (tmp_path/"manifest.csv").write_text("tampered")
    with pytest.raises(ValueError,match="manifest"):module.manifest(tmp_path)
    assert (tmp_path/"manifest_lock.json").read_bytes()==locked


def test_protocol_precedes_actual_generation_and_is_immutable():
    from inverpinn.data.final_confirmatory import verify
    from datetime import datetime
    root=ROOT/"results/final_confirmatory"
    if not root.exists():pytest.skip("Preregistration not executed in this checkout.")
    config,plan,lock=verify(root)
    before=datetime.fromisoformat(lock["frozen_at_utc"]).timestamp()
    assert config["advancement"]==read_config("final_confirmatory.yaml")["advancement"]
    for path in (root/"datasets").glob("*/generation.json"):
        assert path.stat().st_mtime>=before


def test_fitting_execution_mutation_stops(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_final_confirmatory as runner
    directory=tmp_path/"fitting_execution";directory.mkdir()
    (directory/"provenance.json").write_text(json.dumps(dict(source_hashes={"src/inverpinn/training/inverse_formulation.py":"changed"})))
    with pytest.raises(ValueError,match="Code changed"):runner.execution_lock(tmp_path)


def test_sparse_adapter_bitwise_equals_frozen_J1(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_final_confirmatory as runner
    from inverpinn.training.single_source import SparseTrainingData,TrainingSettings
    j=copy.deepcopy(exact_J1());j["recipe"]["training"].update(width=8,depth=2,steps=3,data_batch=6,collocation_batch=12,initial_batch=4,boundary_per_edge=2)
    j["candidate"]["total_updates"]=3;j["profiling"]["terminal_fitting_points"]=16;j["gradient_interval"]=1
    inputs=dict(positions=torch.tensor([[.2,.4],[.6,.7]]),times=torch.tensor([0.,.4,.8]),measurements=torch.tensor([[0.,0.],[.1,.2],[.2,.3]]),
        physics=dict(transport=dict(u=.35,v=-.15,D=.005)))
    estimate=runner.fit_sparse("J1",inputs,j,{},tmp_path,{})
    recipe=j["recipe"];data=SparseTrainingData(inputs["positions"],inputs["times"],inputs["measurements"],**inputs["physics"]["transport"])
    model,source,_,_=fit_profiled(data,TrainingSettings.from_mapping(recipe["training"]),.08,42,recipe["formulation"],j["candidate"],j["profiling"])
    saved=torch.load(tmp_path/"model.pt",weights_only=True)
    for k,v in model.state_dict().items():assert torch.equal(v,saved["model_state_dict"][k])
    for k,v in source.state_dict().items():assert torch.equal(v,saved["source_state_dict"][k])
    assert estimate==source.estimates()


def test_robustness_rechecks_gates_and_exact_model(tmp_path,monkeypatch):
    import inverpinn.data.final_robustness as module
    from inverpinn.data.synthetic_reference import sha256
    config=read_config("final_confirmatory.yaml");frame=fixture_rows()
    for m in METHODS:frame[frame.method==m].to_csv(tmp_path/f"{m}_raw.csv",index=False)
    hashes={f"{m}_raw.csv":sha256(tmp_path/f"{m}_raw.csv") for m in METHODS}
    (tmp_path/"decision.json").write_text(json.dumps(dict(passed=True,result_hashes=hashes)))
    (tmp_path/"configs").mkdir();final=tmp_path/"configs/InverPINN_final.yaml";final.write_text("exact frozen fixture")
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"verify",lambda root:(config,[],{"J1_hash":sha256(final)}))
    assert module.require_confirmation(tmp_path)==config
    monkeypatch.setattr(module,"verify",lambda root:(config,[],{"J1_hash":"wrong"}))
    with pytest.raises(ValueError,match="byte-identical"):module.require_confirmation(tmp_path)
    frame.loc[frame.method=="J1","recovery_success"]=False
    frame[frame.method=="J1"].to_csv(tmp_path/"J1_raw.csv",index=False)
    hashes["J1_raw.csv"]=sha256(tmp_path/"J1_raw.csv")
    (tmp_path/"decision.json").write_text(json.dumps(dict(passed=True,result_hashes=hashes)))
    with pytest.raises(PermissionError,match="gates"):module.require_confirmation(tmp_path)


def test_three_method_end_to_end_and_failed_fit_retention(tmp_path,monkeypatch):
    """Tiny integration fixture, not a final-blind scenario or tuning run."""
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_final_confirmatory as runner
    from inverpinn.data.synthetic_reference import sha256
    from inverpinn.data.single_source_benchmark import object_hash
    from inverpinn.physics.finite_difference import solve_advection_diffusion
    from inverpinn.evaluation.observability import forward
    import yaml
    config=read_config("final_confirmatory.yaml");config["inference_solver"]=dict(nx=21,ny=21,dt=.001,steps=800)
    config["diagnostics"].update(interior_count=16,initial_count=8,boundary_per_edge=4);config["plot_dpi"]=40
    j=copy.deepcopy(exact_J1());j["recipe"]["training"].update(width=8,depth=2,steps=3,data_batch=6,collocation_batch=12,initial_batch=4,boundary_per_edge=2)
    j["candidate"]["total_updates"]=3;j["profiling"]["terminal_fitting_points"]=16;j["gradient_interval"]=1
    (tmp_path/"protocol.yaml").write_text(yaml.safe_dump(config));(tmp_path/"J1_frozen.yaml").write_text(yaml.safe_dump(j))
    monkeypatch.setattr(runner,"install_guard",lambda root:dict(fitting_case=None,method=None))
    identifier=IDS[0];case=tmp_path/"datasets"/identifier;(case/"inputs").mkdir(parents=True);(case/"truth").mkdir()
    truth=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08);positions=torch.tensor([[.2,.4],[.6,.7]],dtype=torch.float64);times=[0.,.4,.8]
    solution=solve_advection_diffusion(**config["inference_solver"],**config["physics"],sources=[truth],output_mode="selected_times",
        output_times=config["evaluation_times"],sensor_positions=positions,measurement_times=times)
    np.savez_compressed(case/"inputs/observations.npz",positions=positions.numpy(),times=times,measurements=solution["measurements"].numpy())
    (case/"inputs/physics.json").write_text(json.dumps(dict(transport=config["physics"],domain=config["domain"],initial_condition="C=0 everywhere",boundary_condition="C=0 on all four edges",units={})))
    (case/"truth/source.json").write_text(json.dumps(truth))
    np.savez_compressed(case/"truth/reference.npz",**{k:v.numpy() for k,v in solution.items()})
    hashes=dict(scenario_id=identifier,observations_hash=sha256(case/"inputs/observations.npz"),physics_hash=sha256(case/"inputs/physics.json"))
    (tmp_path/"input_manifest.json").write_text(json.dumps([hashes]))
    (case/"generation.json").write_text(json.dumps(hashes|dict(source_hash=object_hash(truth),reference_path=f"datasets/{identifier}/truth/reference.npz",
        reference_data_hash=sha256(case/"truth/reference.npz"),reference_target_met=True,max_reference_error_estimate=.001)))
    (tmp_path/"maps").mkdir();locations=np.array([(x,y) for y in (.3,.5,.7) for x in (.3,.5,.7)])
    h=[forward([*p,1.],positions.numpy(),times,config["physics"],config["inference_solver"],.08) for p in locations]
    np.savez_compressed(tmp_path/"maps/unit_responses.npz",locations=locations,unit_readings=h)
    for method in METHODS:
        runner.fit_one(tmp_path,method,identifier)
        row=json.loads((tmp_path/"runs"/method/identifier/"metrics.json").read_text())
        assert row["evaluation_success"] and row["observations_hash"]==hashes["observations_hash"]
        assert row["negative_fraction"]==0
        if method!="classical":assert row["bc_maxabs"]==row["ic_maxabs"]==0
        assert "preserved" in runner.fit_one(tmp_path,method,identifier)
    # A second tiny fixture ID checks terminal failure recording, never retry.
    import shutil
    shutil.copytree(case,tmp_path/"datasets"/IDS[1]);h2=hashes|dict(scenario_id=IDS[1])
    (tmp_path/"input_manifest.json").write_text(json.dumps([hashes,h2]))
    def fail(*args,**kwargs):raise FloatingPointError("deliberate test failure")
    monkeypatch.setattr(runner,"fit_sparse",fail)
    runner.fit_one(tmp_path,"J1",IDS[1])
    row=json.loads((tmp_path/"runs/J1"/IDS[1]/"metrics.json").read_text())
    assert not row["computational_success"] and not row["recovery_success"] and row["training_terminated"]
    assert "deliberate test failure" in row["failure"]


def test_reporting_failure_does_not_promote_or_drop_cases(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import report_final_confirmatory as reporter
    from inverpinn.data.synthetic_reference import sha256
    config=read_config("final_confirmatory.yaml");frame=fixture_rows()
    frame["x_true"]=.3;frame["y_true"]=.7;frame["Q_true"]=1.;frame["x_pred"]=.31;frame["y_pred"]=.7;frame["Q_pred"]=1.02
    frame["reference_target_met"]=True;frame["reference_error_estimate"]=.003
    frame.loc[(frame.method=="J1")&frame.scenario_id.isin(IDS[:4]),"recovery_success"]=False
    frame.loc[0,"computational_success"]=False;frame.loc[0,"evaluation_success"]=False
    for m in METHODS:frame[frame.method==m].to_csv(tmp_path/f"{m}_raw.csv",index=False)
    hashes={f"{m}_raw.csv":sha256(tmp_path/f"{m}_raw.csv") for m in METHODS}
    (tmp_path/"integrity_audit.json").write_text(json.dumps(dict(audited_runs=90,execution_code_unchanged=True,result_hashes=hashes,historical_files_unchanged=7)))
    (tmp_path/"consumed_hashes.json").write_text("{}")
    pd.DataFrame([dict(scenario_id=k,information_group="lower" if i<15 else "higher",sensor_rms=.01,sigma_min=.1) for i,k in enumerate(IDS)]).to_csv(tmp_path/"observability.csv",index=False)
    (tmp_path/"maps").mkdir();(tmp_path/"maps/bank_lock.json").write_text("{}")
    (tmp_path/"fitting_execution").mkdir();(tmp_path/"fitting_execution/source_snapshot.zip").write_bytes(b"fixture")
    monkeypatch.setattr(reporter,"ROOT",tmp_path);monkeypatch.setattr(reporter,"verify",lambda root:(config,[],dict(protocol_hash="fixture",J1_hash="fixture",B_revised_hash="fixture")))
    monkeypatch.setattr(reporter,"manifest",lambda root:[]);monkeypatch.setattr(reporter,"old_inventory",lambda:{})
    monkeypatch.setattr(reporter,"snapshot",lambda root:None);monkeypatch.setattr(reporter,"figures",lambda *args:None)
    reporter.report(tmp_path)
    decision=json.loads((tmp_path/"decision.json").read_text());summary=json.loads((tmp_path/"summary.json").read_text())
    assert not decision["passed"] and not summary["InverPINN_final_created"]
    assert summary["methods"]["J1"]["count"]==30 and summary["methods"]["J1"]["recovered"]==26
    assert len(pd.read_csv(tmp_path/"paired_J1_vs_Brevised.csv"))==30
    assert not (tmp_path/"configs/InverPINN_final.yaml").exists()
    with pytest.raises(FileExistsError):reporter.report(tmp_path)


def test_partial_reporting_is_preserved(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import report_final_confirmatory as reporter
    monkeypatch.setattr(reporter,"verify",lambda root:({},[],{}));monkeypatch.setattr(reporter,"manifest",lambda root:[])
    path=tmp_path/"summary.csv";path.write_text("partial artifact")
    with pytest.raises(FileExistsError,match="Partial"):reporter.report(tmp_path)
    assert path.read_text()=="partial artifact"


def test_robustness_runner_cannot_generate_without_pass(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_final_robustness as runner
    def deny(root):raise PermissionError("confirmation not passed")
    monkeypatch.setattr(runner,"require_confirmation",deny)
    with pytest.raises(PermissionError):runner.initialize(tmp_path/"final",tmp_path/"robust")
    assert not (tmp_path/"robust").exists()


def test_robustness_input_transforms_leave_recipe_and_control_unchanged(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    from run_final_robustness import perturb,CONDITIONS
    positions=torch.tensor(json.loads((ROOT/"configs/single_source_sensors.json").read_text()),dtype=torch.float64)
    inputs=dict(positions=positions,times=torch.tensor([0.,.4,.8]),measurements=torch.arange(60.).reshape(3,20).double(),physics=dict(transport=dict(u=.35,v=-.15,D=.005)))
    config=read_config("final_confirmatory.yaml");spec=config["conditional_robustness"]
    indices={str(k):sorted(v) for k,v in nested_sensors(positions.numpy()).items()};before=copy.deepcopy(inputs);j=exact_J1()
    control=perturb(inputs,"control",spec,indices,0)
    for key in ("positions","times","measurements"):assert torch.equal(control[key],inputs[key])
    for condition in CONDITIONS:
        transformed=perturb(inputs,condition,spec,indices,0)
        assert len(transformed["positions"])==(int(condition.split('_')[1]) if condition.startswith("sensors_") else 20)
        assert transformed["physics"]["transport"]["D"]==.005
    assert inputs["physics"]==before["physics"]
    for key in ("positions","times","measurements"):assert torch.equal(inputs[key],before[key])
    assert exact_J1()==j
