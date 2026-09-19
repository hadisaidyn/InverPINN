"""Reference selection, inverse-crime separation, provenance and input isolation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from inverpinn.data.synthetic_reference import inference_observations, load_model_inputs, observation_coordinates, validate_separation
from inverpinn.evaluation.reference import choose_reference, compare_references
from inverpinn.physics.finite_difference import solve_advection_diffusion


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/"scripts"/f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def artificial_references():
    """Known C_h=C_exact*(1+h): Richardson must recover p=1 and error=h."""
    cases = []
    for n in [5, 9, 17]:
        axis = torch.linspace(0, 1, n, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        exact = (1+x+y)[None]*torch.tensor([1, 2], dtype=torch.float64)[:, None, None]
        cases.append(dict(fields=exact*(1+1/(n-1)), field_times=torch.tensor([0.05, 0.1]),
            measurements=torch.tensor([[0, 0], [0.2, 0.5]], dtype=torch.float64)*(1+1/(n-1)),
            measurement_times=torch.tensor([0, 0.1]), sensor_positions=torch.tensor([[0.37, 0.43], [0.7, 0.8]])))
    return cases


def test_richardson_estimates_are_correct_on_known_power_law():
    rows = compare_references(artificial_references())
    for row in rows:
        if row["scope"] == "native_final_difference":
            assert row["Richardson_relative_L2_estimate"] is None
        else:
            assert row["Richardson_order"] == pytest.approx(1)
            assert row["Richardson_relative_L2_estimate"] == pytest.approx(1/(row["grid_size"]-1))
    decision = choose_reference(rows, {5: 1, 9: 2, 17: 3}, 0.07)
    assert decision["selected_grid"] == 17
    assert not decision["certified_bound"]


def test_nonconvergent_or_zero_differences_cannot_pass_accuracy_target():
    cases = artificial_references()
    for case in cases:
        case["fields"].zero_()
        case["measurements"].zero_()
    decision = choose_reference(compare_references(cases), {5: 1, 9: 2, 17: 3}, 0.01)
    assert decision["selected_grid"] is None
    assert not any(v["meets_estimated_target"] for v in decision["per_grid"].values())


def test_different_observation_designs_cannot_be_compared_as_grid_refinement():
    cases = artificial_references()
    cases[1]["sensor_positions"] += 0.01
    with pytest.raises(ValueError, match="identical sensor_positions"):
        compare_references(cases)


def test_separate_inference_operator_uses_same_physical_points_but_distinct_discretization():
    physics = dict(u=0.35, v=-0.15, D=0.005)
    source = dict(x_s=0.3, y_s=0.7, Q=1, sigma=0.08)
    reference = dict(nx=21, ny=21, dt=0.005, steps=20)
    inference = yaml.safe_load(Path("configs/reference_inference.yaml").read_text())
    inference["solver"] = dict(nx=13, ny=13, dt=0.004, steps=25)
    p = torch.tensor([[0.371, 0.431], [0.311, 0.697]], dtype=torch.float64)
    t = torch.tensor([0, 0.013, 0.067, 0.1], dtype=torch.float64)
    observed = solve_advection_diffusion(**reference, **physics, sources=[source], output_mode="sensor_only",
                                         sensor_positions=p, measurement_times=t.tolist())["measurements"]
    inputs = dict(positions=p, times=t, physics=dict(transport=physics, domain=inference["domain"]))
    predicted = inference_observations(source, inputs, inference, reference)
    assert predicted.shape == observed.shape
    assert float((predicted-observed).abs().max()) > 1e-5
    assert inputs["positions"].tolist() == p.tolist()  # no snapping/mutation
    same = inference | dict(solver=reference)
    with pytest.raises(ValueError, match="grids must differ"):
        validate_separation(reference, same, physics, inference["domain"])
    contaminated = inference | dict(solver=inference["solver"] | dict(sources=[source]))
    with pytest.raises(ValueError, match="no sources"):
        validate_separation(reference, contaminated, physics, inference["domain"])


def small_config(tmp_path):
    base = yaml.safe_load(Path("configs/reference_benchmark.yaml").read_text())
    base["candidates"] = [dict(nx=n, ny=n, dt=0.004/factor**2, steps=25*factor**2)
                          for n,factor in [(11,1), (21,2), (41,4)]]
    base["evaluation_times"] = [0.03, 0.1]
    base["measurement_times"] = dict(start=0.0, stop=0.1, count=7)
    base["sensors"] = dict(count=5)
    inference = yaml.safe_load(Path("configs/reference_inference.yaml").read_text())
    inference["solver"] = dict(nx=17, ny=17, dt=0.004, steps=25)
    inference_path = tmp_path/"inference.yaml"
    inference_path.write_text(yaml.safe_dump(inference))
    base["inference_config"] = str(inference_path)
    base_path = tmp_path/"benchmark.yaml"
    base_path.write_text(yaml.safe_dump(base))
    config = dict(benchmark_config=str(base_path), noise=dict(kind="gaussian_absolute", std=0.005),
                  scenarios=[dict(scenario_id="pilot_test", seed=17, source=base["source"])])
    return config, base


def test_pilot_determinism_metadata_configuration_and_leakage_guard(tmp_path, monkeypatch):
    config, base = small_config(tmp_path)
    generator = module("generate_reference_pilot")
    before = json.loads(json.dumps(config))
    outputs = [generator.generate_pilot(config, tmp_path/name)/"pilot_test" for name in ("first", "second")]
    assert config == before
    with np.load(outputs[0]/"inputs/observations.npz") as first, np.load(outputs[1]/"inputs/observations.npz") as second:
        assert set(first.files) == {"positions", "times", "measurements"}
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])
    scenario = outputs[0]
    metadata = json.loads((scenario/"metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["random_seed"] == 17 and metadata["noise_seed"] == 18
    assert metadata["provenance"]["git_commit"]
    assert metadata["generated_at_utc"].endswith("+00:00")
    assert metadata["reference_accuracy"]["certified_bound"] is False
    assert metadata["benchmark_release_ready"] is False
    source = json.loads((scenario/metadata["source_truth_file"]).read_text())
    assert all(source[k] == base["source"][k] for k in base["source"])
    chosen = metadata["reference_solver"]
    assert chosen["dt"]*chosen["steps"] == pytest.approx(0.1)
    assert chosen["hypothetical_full_history_bytes"] > chosen["retained_field_bytes"]
    assert chosen["peak_process_rss_bytes"] > 0
    with np.load(scenario/metadata["reference_checkpoint"]) as truth:
        assert truth["fields"].shape == (2, chosen["ny"], chosen["nx"])
        assert truth["measurements"].shape == (7, 5)
        assert truth["field_times"].tolist() == base["evaluation_times"]
    restored = yaml.safe_load((scenario/"config.yaml").read_text())
    assert restored["reference_configuration"]["candidates"] == base["candidates"]
    assert yaml.safe_load(yaml.safe_dump(restored)) == restored
    # Enforce that the loader opens only inputs/, even though source truth and
    # generation config exist beside it. Poisoning truth does not affect inputs.
    original_load = np.load
    def guard(path, **kwargs):
        assert Path(path).parent == scenario/"inputs"
        return original_load(path, **kwargs)
    monkeypatch.setattr(np, "load", guard)
    model_inputs = load_model_inputs(scenario)
    assert model_inputs["features"].shape == (35, 3)
    assert set(model_inputs) == {"positions", "times", "measurements", "features", "targets", "physics"}
    torch.testing.assert_close(model_inputs["features"][:, :2], model_inputs["positions"].repeat(7, 1))
    torch.testing.assert_close(model_inputs["features"][:, 2], model_inputs["times"].repeat_interleave(5))
    assert set(model_inputs["physics"]["transport"]) == {"u", "v", "D"}
    digest = (scenario/"inputs/observations.npz").read_bytes()
    with pytest.raises(FileExistsError):
        generator.generate_pilot(config, tmp_path/"first")
    assert (scenario/"inputs/observations.npz").read_bytes() == digest


def test_input_archive_rejects_source_truth(tmp_path):
    directory = tmp_path/"inputs"
    directory.mkdir()
    np.savez(directory/"observations.npz", positions=np.zeros((2,2)), times=np.zeros(1),
             measurements=np.zeros((1,2)), source_positions=np.array([[0.3,0.7]]))
    with pytest.raises(ValueError, match="truth is forbidden"):
        load_model_inputs(tmp_path)


def test_placement_is_independent_of_source_and_global_rng(tmp_path):
    _, config = small_config(tmp_path)
    state = torch.random.get_rng_state().clone()
    a, times = observation_coordinates(config)
    b, _ = observation_coordinates(config | dict(source=dict(x_s=0.8, y_s=0.2, Q=2, sigma=0.08)))
    assert torch.equal(a, b) and torch.equal(state, torch.random.get_rng_state())
    c, _ = observation_coordinates(config | dict(seed=config["seed"]+1))
    assert not torch.equal(a, c)


def test_storage_equivalence_report_contains_measured_zero_differences():
    report = module("benchmark_reference").storage_equivalence()
    assert all(v == 0.0 for k,v in report.items() if k.endswith("difference"))
