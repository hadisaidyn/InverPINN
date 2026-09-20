"""Diagnostic isolation, numerical linearity and local-observability contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inverpinn.data import diagnostic_v2 as data
from inverpinn.data.fresh_blind import fresh_plan, REVISED_HASH
from inverpinn.data.single_source_benchmark import scenario_plan
from inverpinn.data.synthetic_reference import ROOT, sha256


@pytest.fixture
def config():
    return yaml.safe_load((ROOT/"configs/observability_diagnosis.yaml").read_text())


def old_configs():
    return [yaml.safe_load((ROOT/"configs"/name).read_text()) for name in
            ("single_source_benchmark.yaml", "fresh_blind_benchmark.yaml")]


def test_new_lhs_has_no_overlap_and_one_point_in_every_marginal_stratum(config):
    legacy, fresh = old_configs()
    rows = data.diagnostic_plan(config, legacy, fresh)
    assert rows == data.diagnostic_plan(config, legacy, fresh)
    old = scenario_plan(legacy)+fresh_plan(fresh, legacy)
    for key in ("scenario_id", "generation_seed", "source_hash"):
        values = {row[key] for row in rows}
        assert len(values)==30 and not values & {row[key] for row in old}
    for key, lower, span in (("x_s",.25,.5),("y_s",.25,.5),("Q",.8,.6)):
        strata = [int(30*(row["source"][key]-lower)/span) for row in rows]
        assert sorted(strata)==list(range(30))


@pytest.mark.parametrize("seed",[1909202601,1909202602,2009202605])
def test_consumed_master_seeds_rejected(config,seed):
    config["scenario_sampling"]["seed"]=seed
    with pytest.raises(ValueError,match="new master"):
        data.diagnostic_plan(config,*old_configs())


@pytest.mark.parametrize("identifier",["development_001","heldout_001","fresh_001","diagnostic_v2_031","../old"])
def test_consumed_and_invalid_ids_rejected(tmp_path,identifier):
    with pytest.raises(ValueError,match="Only diagnostic"):
        data.diagnostic_path(tmp_path,identifier)


def test_fitting_guard_blocks_truth_oracles_and_consumed_results():
    root=ROOT/"results/observability_diagnosis"
    for path in (ROOT/"results/fresh_blind_benchmark/B_revised_raw.csv",root/"conditional_Q_results.csv",
                 root/"diagnostic_manifest.csv",root/"datasets/diagnostic_v2_001/truth/source.json",
                 root/"datasets/diagnostic_v2_002/inputs/observations.npz"):
        with pytest.raises(PermissionError):
            data.check_open(root,path,"diagnostic_v2_001")
    data.check_open(root,root/"datasets/diagnostic_v2_001/inputs/observations.npz","diagnostic_v2_001")
    data.check_open(root,root/"runs/pinn/diagnostic_v2_001/training_history.jsonl","diagnostic_v2_001")


def test_unchanged_physics_noise_geometry_model_and_recovery(config):
    _,fresh=old_configs()
    for key in ("physics","source_family","sensors_file","sensor_count","noise_std","measurement_times","reference","recovery"):
        assert config[key]==fresh[key]
    assert sha256(ROOT/"configs/B_revised.yaml")==REVISED_HASH
    assert config["optimization"]["primary_seed"]==42
    assert config["observability_map"]["representative_Q"]==1.1


def test_pinn_waits_for_immutable_information_stage(tmp_path):
    with pytest.raises(RuntimeError,match="before PINN"):
        data.require_information_complete(tmp_path)
    result=tmp_path/"diagnostic.csv"
    result.write_text("fixture")
    (tmp_path/"information_complete.json").write_text(json.dumps({"hashes":{"diagnostic.csv":sha256(result)}}))
    data.require_information_complete(tmp_path)
    result.write_text("changed")
    with pytest.raises(ValueError,match="artifact changed"):
        data.require_information_complete(tmp_path)


def test_existing_diagnostic_set_is_never_overwritten(config,tmp_path):
    with pytest.raises(FileExistsError,match="immutable"):
        data.freeze(config,tmp_path)


def test_analytic_Q_matches_numerical_optimizer_and_positive_boundary():
    from inverpinn.evaluation.observability import conditional_Q
    h=np.array([0.,.1,.4,1.,2.]); y=1.27*h
    result=conditional_Q(h,y)
    assert result["Q_analytic"]==pytest.approx(1.27,abs=1e-13)
    assert result["Q_numerical"]==pytest.approx(1.27,abs=1e-9)
    assert result["sensor_mse"]<1e-28 and result["numerical_success"]
    # Positive domain's zero infimum, not a PINN parameter clipping operation.
    assert conditional_Q(h,-h)["Q_analytic"]==0
    with pytest.raises(ValueError,match="Zero source"):
        conditional_Q(np.zeros(5),y)


def test_amplitude_direction_and_shape_difference_are_distinct():
    from inverpinn.evaluation.observability import amplitude_mismatch,conditional_Q
    reference=np.array([1.,2.,3.])
    strong=1.2*reference
    result=amplitude_mismatch(strong,reference)
    assert result["amplitude_ratio"]==pytest.approx(1.2)
    assert result["signed_observation_bias"]>0
    assert conditional_Q(strong,reference)["Q_analytic"]==pytest.approx(1/1.2)
    assert amplitude_mismatch(.8*reference,reference)["signed_observation_bias"]<0
    different=np.array([2.,1.,3.])
    assert amplitude_mismatch(different,reference)["relative_observation_l2"]>0


def test_scaled_J_matches_analytic_derivatives_and_fixed_step_stability(config):
    from inverpinn.evaluation.observability import scaled_jacobian,jacobian_stability
    a=np.array([.2,.5,1.,1.5]);b=np.array([1.,-.4,.3,.2])
    response=lambda p:np.exp(p[0]*a+p[1]*b)
    theta=np.array([.4,.6,1.2]); scales=config["scaling"]["parameter_scales"]
    h,jac=scaled_jacobian(theta,response,scales,.88,.00025)
    expected=np.column_stack([theta[2]*a*h,theta[2]*b*h,h])*scales/.88
    np.testing.assert_allclose(jac,expected,rtol=1e-8,atol=1e-11)
    selected,rows=jacobian_stability(theta,response,config["scaling"],config["sensitivity"])
    np.testing.assert_array_equal(selected,jac)
    assert all(r["stable"] for r in rows)
    assert config["scaling"]["parameter_scales"]==[.5,.5,.6]


def test_information_svd_covariance_and_rank_deficiency():
    from inverpinn.evaluation.observability import information_metrics
    jac=np.diag([4.,2.,1.])
    m=information_metrics(jac,[.5,.5,.6])
    assert m["rank"]==3 and m["jacobian_condition"]==4
    assert m["information_condition"]==16 and m["information_eigenvalue_min"]==1
    assert m["information_logdet"]==pytest.approx(np.log(64))
    assert m["normalized_x_std_proxy"]==.25
    assert m["physical_Q_std_proxy"]==.6 and m["Q_effective_information"]==1
    singular=information_metrics(np.ones((8,3)),[.5,.5,.6])
    assert singular["rank"]==1 and singular["jacobian_condition"] is None
    assert singular["physical_Q_std_proxy"] is None
    assert singular["Q_effective_information"]<1e-28


def test_geometry_equations_known_wind_and_no_downwind_sensor():
    from inverpinn.evaluation.observability import geometry
    physics=dict(u=1.,v=0.,D=0.)
    m=geometry([.5,.5],[[.6,.5],[.7,.8],[.4,.5]],physics,.08,.8)
    assert m["downwind_sensor_count"]==2 and m["plume_corridor_count"]==1
    assert m["nearest_downwind_distance"]==pytest.approx(.1)
    assert m["min_downwind_crosswind_distance"]==0
    empty=geometry([.5,.5],[[.1,.1]],physics,.08,.8)
    assert empty["nearest_downwind_distance"] is None and empty["downwind_sensor_count"]==0


def test_envelope_inversion_and_gain_no_epsilon_at_edges():
    from inverpinn.evaluation.observability import envelope_latent
    axis=np.linspace(0,1,11);yy,xx=np.meshgrid(axis,axis,indexing="ij")
    envelope=16*xx*(1-xx)*yy*(1-yy)
    fields=np.array([2*envelope])
    rows,maps=envelope_latent(fields,axis,axis,[.8],.8,.01)
    np.testing.assert_allclose(maps[0]["positive_latent"][1:-1,1:-1],2)
    expected=2+np.log(-np.expm1(-2))
    assert rows[0]["required_raw_latent_median"]==pytest.approx(expected)
    assert rows[0]["excluded_zero_envelope"]==40
    assert np.isnan(maps[0]["positive_latent"][0]).all()
    np.testing.assert_allclose(maps[0]["transform_gain"][1:-1,1:-1],envelope[1:-1,1:-1]*(1-np.exp(-2)))


def small_operator():
    positions=np.array([[.2,.35],[.4,.4],[.6,.65],[.8,.5],[.45,.7]])
    times=np.array([0.,.02,.04,.08])
    return positions,times,dict(u=.35,v=-.15,D=.005),dict(nx=21,ny=21,dt=.001,steps=80)


def test_actual_forward_linearity_mapping_and_resolution_schedule():
    import torch
    from inverpinn.evaluation.observability import forward,conditional_Q
    from inverpinn.physics.finite_difference import solve_advection_diffusion
    torch.set_num_threads(1)
    positions,times,physics,solver=small_operator()
    theta=[.47,.55,1.23]
    h=forward([*theta[:2],1.],positions,times,physics,solver,.08)
    y=forward(theta,positions,times,physics,solver,.08)
    np.testing.assert_allclose(y,theta[2]*h,rtol=3e-15,atol=1e-16)
    assert conditional_Q(h,y)["Q_analytic"]==pytest.approx(theta[2],abs=1e-13)
    for grid in [21,31]:
        out=solve_advection_diffusion(**(solver|dict(nx=grid,ny=grid)),**physics,
            sources=[dict(x_s=theta[0],y_s=theta[1],Q=theta[2],sigma=.08)],
            output_mode="sensor_only",sensor_positions=torch.tensor(positions),measurement_times=times)
        np.testing.assert_array_equal(out["sensor_positions"].numpy(),positions)
        np.testing.assert_array_equal(out["measurement_times"].numpy(),times)
        np.testing.assert_array_equal(forward(theta,positions,times,physics,solver|dict(nx=grid,ny=grid),.08),out["measurements"].numpy().reshape(-1))


def test_actual_forward_J_step_stability(config):
    from inverpinn.evaluation.observability import unit_operator,jacobian_stability
    positions,times,physics,solver=small_operator()
    response=unit_operator(positions,times,physics,solver,.08)
    jac,rows=jacobian_stability([.47,.55,1.23],response,config["scaling"],config["sensitivity"])
    assert jac.shape==(20,3) and all(r["stable"] for r in rows)


def test_classical_fitting_uses_only_observations_and_observation_objective(config):
    import inspect
    from inverpinn.evaluation.observability import forward,classical_inverse,unit_operator,select_starts
    positions,times,physics,solver=small_operator()
    theta=[.47,.55,1.23]
    y=forward(theta,positions,times,physics,solver,.12)
    candidates=np.array([[x,v] for x in [.3,.5,.7] for v in [.3,.5,.7]])
    op=unit_operator(positions,times,physics,solver,.12)
    responses=np.stack([op(p) for p in candidates])
    best,records=classical_inverse(y,positions,times,physics,solver,.12,candidates,responses,
        config["classical"],config["scaling"],config["sensitivity"])
    assert len(records)==3 and best["sensor_mse"]==min(r["sensor_mse"] for r in records)
    np.testing.assert_allclose([best["x_pred"],best["y_pred"],best["Q_pred"]],theta,atol=1e-5)
    assert not {"truth","source_truth","scenario_path","metadata"} & set(inspect.signature(classical_inverse).parameters)
    initial,indices=select_starts(candidates,responses,y,3,.08)
    np.testing.assert_array_equal(initial,candidates[indices])


def test_observability_map_interface_has_no_hidden_truth():
    import inspect
    from inverpinn.evaluation.observability import scaled_jacobian,forward
    assert set(inspect.signature(scaled_jacobian).parameters)=={"theta","response","parameter_scales","concentration_scale","step"}
    assert not {"truth","scenario","dataset","reference"} & set(inspect.signature(forward).parameters)


def test_consumed_inventory_change_detected_without_parsing_outcomes(tmp_path,monkeypatch):
    monkeypatch.setattr(data,"consumed_inventory",lambda:{"old/file":"hash"})
    (tmp_path/"consumed_hashes.json").write_text(json.dumps({"old/file":"hash"}))
    assert data.verify_consumed(tmp_path)==1
    monkeypatch.setattr(data,"consumed_inventory",lambda:{"old/file":"changed"})
    with pytest.raises(ValueError,match="Consumed"):
        data.verify_consumed(tmp_path)


def test_network_envelope_is_read_only_and_gradients_are_analytical():
    import torch
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.evaluation.observability import network_envelope
    model=ConstrainedConcentrationMLP(.8,width=4,depth=1,constraint="hard")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.fill_(1.)
    before={k:v.clone() for k,v in model.state_dict().items()}
    axis=np.linspace(0,1,11);yy,xx=np.meshgrid(axis,axis,indexing="ij")
    E=16*xx*(1-xx)*yy*(1-yy)
    rows,maps=network_envelope(model,np.array([E]),axis,[.8],.8,.01)
    np.testing.assert_allclose(maps[0]["raw_latent"],1.,atol=1e-7)
    np.testing.assert_allclose(maps[0]["transform_gain"],E/(1+np.exp(-1)),atol=2e-7)
    np.testing.assert_allclose(maps[0]["raw_spatial_gradient"],0.,atol=1e-8)
    np.testing.assert_allclose(maps[0]["C_time_gradient"],E*np.log1p(np.e)/.8,atol=4e-7)
    assert len(rows)==2 and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_live_audit_guard_blocks_truth_inside_child_process(tmp_path):
    import subprocess
    import sys
    case=tmp_path/"datasets/diagnostic_v2_001/truth"
    case.mkdir(parents=True);(case/"source.json").write_text("{}");
    program="""
from pathlib import Path
from inverpinn.data.diagnostic_v2 import install_guard
import sys
root=Path(sys.argv[1])
state=install_guard(root);state['fitting_case']='diagnostic_v2_001'
try:
    (root/'datasets/diagnostic_v2_001/truth/source.json').read_text()
except PermissionError:
    pass
else:
    raise AssertionError('Truth was readable during fitting')
state['fitting_case']=None
assert (root/'datasets/diagnostic_v2_001/truth/source.json').read_text()=='{}'
"""
    result=subprocess.run([sys.executable,"-c",program,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_failure_labels_do_not_turn_weak_information_into_causal_proof(config):
    from inverpinn.evaluation.observability import failure_evidence
    row=dict(classical_recovery_success=True,pinn_recovery_success=False,sigma_min=.01,Q_effective_information=.001,
        classical_localization_error=.01,pinn_localization_error=.2,classical_relative_strength_error=.01,
        pinn_relative_strength_error=.4,conditional_relative_Q_error=.01,conditional_321_relative_Q_error=.005)
    thresholds=dict(sigma_min=.1,Q_effective_information=.01)
    result=failure_evidence(row,thresholds,config["recovery"])
    assert result["flag_A"] and result["primary_evidence_attribution"]=="PINN optimization/model limitation"
    assert result["relatively_weak_information"] and not result["attribution_is_causal_proof"]
    row["classical_recovery_success"]=False
    result=failure_evidence(row,thresholds,config["recovery"])
    assert result["flag_B"] and result["primary_evidence_attribution"]=="mixed/uncertain"
    row.update(pinn_recovery_success=True,conditional_relative_Q_error=.3)
    assert failure_evidence(row,thresholds,config["recovery"])["primary_evidence_attribution"]=="numerical forward-model mismatch"


def test_manifest_rejects_tampering_without_overwriting(tmp_path,monkeypatch):
    from inverpinn.data.single_source_benchmark import object_hash
    source=dict(x_s=.3,y_s=.7,Q=1.,sigma=.08)
    plan=[dict(scenario_id="diagnostic_v2_001",source=source)]
    monkeypatch.setattr(data,"verify",lambda root:({},plan,{"protocol_hash":"fixture"}))
    case=tmp_path/"datasets/diagnostic_v2_001"
    (case/"inputs").mkdir(parents=True);(case/"truth").mkdir()
    for name in ("inputs/observations.npz","inputs/physics.json","truth/reference.npz"):
        (case/name).write_bytes(b"fixture")
    row=dict(scenario_id="diagnostic_v2_001",max_reference_error_estimate=.001,
        observations_hash=sha256(case/"inputs/observations.npz"),physics_hash=sha256(case/"inputs/physics.json"),
        reference_data_hash=sha256(case/"truth/reference.npz"),reference_path="datasets/diagnostic_v2_001/truth/reference.npz")
    (case/"generation.json").write_text(json.dumps(row))
    first=data.manifest(tmp_path)
    assert data.manifest(tmp_path)==first
    frozen=(tmp_path/"manifest_lock.json").read_bytes()
    (tmp_path/"diagnostic_manifest.csv").write_text("tampered")
    with pytest.raises(ValueError,match="Immutable"):
        data.manifest(tmp_path)
    assert (tmp_path/"manifest_lock.json").read_bytes()==frozen


def test_report_pairs_all_cases_and_keeps_failed_cases(monkeypatch):
    import sys
    import pandas as pd
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    from report_observability_diagnosis import paired_inputs,signed_strength
    frame=pd.DataFrame(dict(scenario_id=data.IDS,observations_hash=[f"hash_{i}" for i in range(30)],
                            recovery_success=[False]*30))
    paired_inputs(frame,frame.copy())
    with pytest.raises(ValueError,match="30 planned"):
        paired_inputs(frame,frame.iloc[:-1])
    changed=frame.copy();changed.loc[3,"observations_hash"]="wrong"
    with pytest.raises(ValueError,match="observations differ"):
        paired_inputs(frame,changed)
    counts=signed_strength(np.array([.9,1.1,1.,np.nan]),np.ones(4))
    assert [counts[k] for k in ("underestimates","overestimates","approximately_equal","missing")]==[1,1,1,1]


def test_information_and_final_plots_accept_real_schema(tmp_path,config):
    import pandas as pd
    from inverpinn.evaluation.observability import envelope_latent,network_envelope
    from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
    from inverpinn.visualization.observability import information_figures,final_figures
    config=copy.deepcopy(config);config["plot_dpi"]=40
    sensors=json.loads((ROOT/config["sensors_file"]).read_text())
    (tmp_path/"sensor_geometry.json").write_text(json.dumps(sensors))
    mapping=pd.DataFrame([dict(x=x,y=y,sensor_rms=.01+x,Q_sensitivity_norm=.1+x,
        x_sensitivity_norm=.2+y,y_sensitivity_norm=.3+x,jacobian_condition=2+x+y,sigma_min=.05+x*y)
        for x in (.25,.5,.75) for y in (.25,.5,.75)])
    identifiers=["diagnostic_v2_001","diagnostic_v2_002"]
    combined=pd.DataFrame(dict(scenario_id=identifiers,sensor_rms=[.01,.1],
        localization_error=[.01,.02],relative_strength_error=[.01,.03],
        jacobian_condition=[2.,4.],sigma_min=[.02,.1],Q_effective_information=[.002,.01]))
    for method in ("classical","pinn"):
        for key in ("localization_error","relative_strength_error","relative_l2"):
            combined[f"{method}_{key}"]=[.01,.02]
    resolutions=pd.DataFrame([dict(scenario_id=k,grid=g,signed_relative_Q_error=-1/g,relative_Q_error=1/g)
                             for k in identifiers for g in (101,201,321)])
    axis=np.linspace(0,1,11);yy,xx=np.meshgrid(axis,axis,indexing="ij")
    fields=np.array([16*xx*(1-xx)*yy*(1-yy)*.1])
    reference,maps=envelope_latent(fields,axis,axis,[.8],.8,.01)
    actual,_=network_envelope(ConstrainedConcentrationMLP(.8,4,1,"hard"),fields,axis,[.8],.8,.01)
    envelope=pd.DataFrame([dict(scenario_id=k,x_true=.4,y_true=.5,**r) for k in identifiers for r in reference])
    network=pd.DataFrame([dict(scenario_id=k,**r) for k in identifiers for r in actual])
    frames=dict(resolution_bias_results=resolutions,envelope_reference=envelope)
    information_figures(tmp_path,config,mapping,frames,combined)
    directory=tmp_path/"diagnostics"/config["envelope"]["latent_map_scenario"];directory.mkdir(parents=True)
    np.savez_compressed(directory/"envelope_maps.npz",**{k:np.stack([m[k] for m in maps]) for k in maps[0]})
    chain=pd.DataFrame(dict(scenario_id=identifiers,Q_true=[1.,1.2],Q_conditional=[.99,1.19],Q_classical=[.98,1.18],Q_PINN=[.9,1.1]))
    categories=pd.DataFrame(dict(evidence_category=["E: both recover","A: classical recovers; PINN fails"]))
    final_figures(tmp_path,config,combined,chain,categories,envelope,network)
    expected=["Q_bias_chain","Q_error_vs_resolution","sensor_signal_map","Q_sensitivity_map","location_sensitivity_map",
        "jacobian_condition_map","observability_map","sensor_signal_vs_error","conditioning_vs_error",
        "classical_vs_PINN","failure_categories","constraint_envelope_diagnostic"]
    assert all((tmp_path/f"{name}.{extension}").stat().st_size>0 for name in expected for extension in ("png","pdf"))
    with pytest.raises(FileExistsError,match="preserved"):
        information_figures(tmp_path,config,mapping,frames,combined)


def test_end_to_end_report_retains_failed_rows_and_writes_finite_json(tmp_path,config,monkeypatch):
    import pandas as pd
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import report_observability_diagnosis as reporting
    import inverpinn.visualization.observability as plotting
    identifiers=list(data.IDS)
    frozen=[dict(scenario_id=k,reference_target_met=True,max_reference_error_estimate=.005) for k in identifiers]
    monkeypatch.setattr(reporting,"verify",lambda root:(config,[],{}))
    monkeypatch.setattr(reporting,"manifest",lambda root:frozen)
    monkeypatch.setattr(reporting,"require_information_complete",lambda root:None)
    monkeypatch.setattr(reporting,"verify_consumed",lambda root:5201)
    monkeypatch.setattr(plotting,"final_figures",lambda *args:None)
    base=pd.DataFrame(dict(scenario_id=identifiers,observations_hash=[f"hash{i}" for i in range(30)],
        computational_success=[True]*30,recovery_success=[True]*30,x_pred=[.4]*30,y_pred=[.5]*30,
        Q_pred=[.99]*30,Q_true=[1.]*30,signed_Q_error=[-.01]*30))
    for key in ("localization_error","relative_strength_error","relative_l2","rmse","mae","sensor_mse"):
        base[key]=.01
    base.to_csv(tmp_path/"classical_inverse_results.csv",index=False)
    pinn=base.copy()
    pinn.loc[0,"computational_success"]=False;pinn.loc[0,"recovery_success"]=False
    for key in ("Q_pred","signed_Q_error","localization_error","relative_strength_error","relative_l2","rmse","mae","sensor_mse"):
        pinn.loc[0,key]=np.nan
    for key in ("pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs","negative_fraction","minimum_concentration"):
        pinn[key]=0.
    pinn.to_csv(tmp_path/"pinn_results.csv",index=False)
    (tmp_path/"pinn_complete.json").write_text(json.dumps(dict(raw_hash=sha256(tmp_path/"pinn_results.csv"))))
    (tmp_path/"information_summary.json").write_text(json.dumps(dict(relative_coverage_thresholds=dict(sigma_min=.1,Q_effective_information=.01))))
    resolutions=pd.DataFrame([dict(scenario_id=k,grid=g,relative_Q_error=.01,signed_relative_Q_error=-.01,
        Q_analytic=.99,Q_true=1.,amplitude_ratio=1.01,relative_observation_l2=.01,signed_observation_bias=.001)
        for k in identifiers for g in (101,201,321)])
    resolutions.to_csv(tmp_path/"resolution_bias_results.csv",index=False)
    resolutions.query("grid == 201").to_csv(tmp_path/"conditional_Q_results.csv",index=False)
    jac=pd.DataFrame(dict(scenario_id=identifiers))
    for key in ("sigma_min","sigma_max","jacobian_condition","x_sensitivity_norm","y_sensitivity_norm",
                "Q_sensitivity_norm","Q_effective_information","physical_x_std_proxy","physical_y_std_proxy","physical_Q_std_proxy"):
        jac[key]=.5
    jac.to_csv(tmp_path/"jacobian_metrics.csv",index=False)
    geo=pd.DataFrame(dict(scenario_id=identifiers))
    for key in ("sensor_rms","maximum_sensor_signal","x_true","y_true","Q_true","nearest_sensor_distance","boundary_distance",
        "downwind_sensor_count","nearest_downwind_distance","min_downwind_crosswind_distance","reachable_downwind_count","plume_corridor_count"):
        geo[key]=.5
    geo.to_csv(tmp_path/"geometry_metrics.csv",index=False)
    pd.DataFrame(dict(x=[.5],y=[.5])).to_csv(tmp_path/"map_metrics.csv",index=False)
    envelope=pd.DataFrame(dict(scenario_id=identifiers,scope=["active"]*30,time=[.8]*30,x_true=[.4]*30,y_true=[.5]*30))
    for key in ("required_positive_latent_q95","required_raw_latent_q95","transform_gain_q05"):
        envelope[key]=.1
    envelope.to_csv(tmp_path/"envelope_reference.csv",index=False)
    network=envelope[["scenario_id","scope","time"]].copy()
    for key in ("positive_latent_q95","raw_latent_q95","transform_gain_q05","raw_spatial_gradient_q95","C_spatial_gradient_q95"):
        network[key]=.1
    network.to_csv(tmp_path/"envelope_network.csv",index=False)
    reporting.report(tmp_path)
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["methods"]["pinn"]["total"]==30
    assert summary["methods"]["pinn"]["recovered"]==29
    assert summary["methods"]["pinn"]["metrics"]["relative_l2"]["missing"]==1
    assert len(pd.read_csv(tmp_path/"failure_decomposition.csv"))==30
    with pytest.raises(FileExistsError,match="preserved"):
        reporting.report(tmp_path)


def test_map_point_runs_without_any_scenario_or_truth_files(tmp_path,config,monkeypatch):
    """Exercise the actual map runner, not just the pure forward signature."""
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_observability_diagnosis as runner
    monkeypatch.setattr(runner,"install_guard",lambda root:None)
    fixture=copy.deepcopy(config)
    fixture["classical"]["grid"]=9
    fixture["inference_grids"]=[dict(nx=9,ny=9,dt=.001,steps=800)]
    (tmp_path/"protocol.yaml").write_text(yaml.safe_dump(fixture))
    (tmp_path/"sensor_geometry.json").write_text((ROOT/config["sensors_file"]).read_text())
    assert not (tmp_path/"datasets").exists()
    result=runner.map_point(tmp_path,0,[.4,.6])
    assert result["x"]==.4 and result["y"]==.6 and result["Q"]==1.1
    assert result["Q_sensitivity_norm"]>0 and result["sensor_rms"]>0
    assert not (tmp_path/"datasets").exists()
    with np.load(tmp_path/"maps/points/point_000/checkpoint.npz",allow_pickle=False) as arrays:
        assert arrays["scaled_jacobian"].shape==(81*20,3)
