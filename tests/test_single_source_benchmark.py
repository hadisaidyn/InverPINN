"""Preregistration, sparse-only training, retained failures and exact metrics."""

from dataclasses import fields
import importlib.util
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from inverpinn.data.single_source_benchmark import (freeze_manifest, freeze_protocol, generate_scenario,
    load_frozen, object_hash, scenario_plan, write_csv)
from inverpinn.data.synthetic_reference import load_model_inputs, sha256
from inverpinn.evaluation.single_source_benchmark import (diagnostic_points, distribution, evaluate_run,
    load_evaluation_truth, parameter_metrics, physical_diagnostics, recovery_decision, summarize, wilson_interval)
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings, fit_sparse_source

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("single_source_runner", ROOT/"scripts/run_single_source_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return yaml.safe_load((ROOT/"configs/single_source_benchmark.yaml").read_text())


def tiny_training():
    settings = config()["training"]
    settings.update(width=8, depth=2, steps=3, data_batch=4, collocation_batch=8,
                    initial_batch=4, boundary_per_edge=2)
    return settings


def sparse_data():
    return SparseTrainingData(torch.tensor([[.3, .7], [.6, .4]]), torch.tensor([0., .1]),
        torch.tensor([[0., 0.], [.04, .02]]), .35, -.15, .005)


def test_frozen_protocol_retains_existing_architecture_and_optimizer():
    frozen = config()
    old = yaml.safe_load((ROOT/"configs/inverse_pinn.yaml").read_text())
    for key in ("width", "depth", "steps", "learning_rate", "loss_weights", "data_batch",
                "collocation_batch", "initial_batch", "boundary_per_edge"):
        assert frozen["training"][key] == old[key]
    assert frozen["reference"]["selected_grid"] == 641
    assert frozen["noise_std"] == 0


def test_scenarios_reproducible_without_overlap_or_global_rng_dependency():
    c = config()
    a = scenario_plan(c)
    np.random.seed(91)
    _ = np.random.uniform(size=71)
    b = scenario_plan(c)
    assert a == b
    dev = [r for r in a if r["split"] == "development"]
    test = [r for r in a if r["split"] == "heldout"]
    assert (len(dev), len(test)) == (10, 30)
    for key in ("scenario_id", "source_hash", "generation_seed"):
        assert set(r[key] for r in dev).isdisjoint(r[key] for r in test)
        assert len({r[key] for r in a}) == 40
    for r in a:
        s = r["source"]
        assert .25 <= s["x_s"] <= .75 and .25 <= s["y_s"] <= .75 and .8 <= s["Q"] <= 1.4


def test_fixed_subset_and_primary_seeds_are_not_outcome_dependent():
    c = config()
    runs = runner().planned_runs(c, scenario_plan(c))
    assert len(runs) == 50 and len(set(runs)) == 50
    assert sum(s.startswith("heldout") and seed == 42 for s, seed in runs) == 30
    assert {s for s, seed in runs if seed != 42} == set(c["optimization"]["additional_seed_scenarios"])


def test_strict_training_records_have_no_truth_or_filesystem_capability():
    assert set(f.name for f in fields(SparseTrainingData)) == {"positions", "times", "measurements", "u", "v", "D"}
    assert set(inspect.signature(fit_sparse_source).parameters) == {
        "data", "settings", "sigma", "optimization_seed", "logger", "on_step"}
    assert not hasattr(sparse_data(), "__dict__")
    with pytest.raises(TypeError):
        TrainingSettings.from_mapping(tiny_training() | {"true_source": {"x_s": .3}})
    with pytest.raises(TypeError):
        fit_sparse_source({"source": [1, 2]}, TrainingSettings.from_mapping(tiny_training()), .08, 42)


def test_sparse_data_detaches_backing_storage():
    original = sparse_data()
    duplicate = SparseTrainingData(original.positions, original.times, original.measurements, .35, -.15, .005)
    original.measurements.fill_(123)
    assert duplicate.measurements.max() < 1


def test_initial_guess_independent_of_scenario_truth_and_optimization_seed(monkeypatch):
    import inverpinn.training.single_source as module
    starts = []
    def capture(*args, **kwargs):
        starts.append(kwargs["source_model"].estimates())
        return None, None, []
    monkeypatch.setattr(module, "train_forward_pinn", capture)
    settings = TrainingSettings.from_mapping(tiny_training())
    for seed in (42, 314, 2718):
        data = sparse_data()
        data.measurements.mul_(seed)
        fit_sparse_source(data, settings, .08, seed)
    assert starts == [dict(x_s=.5, y_s=.5, Q=.5)]*3


def test_fits_are_bitwise_deterministic_with_identical_sparse_inputs():
    settings = TrainingSettings.from_mapping(tiny_training())
    a = fit_sparse_source(sparse_data(), settings, .08, 42)
    b = fit_sparse_source(sparse_data(), settings, .08, 42)
    for component in (0, 1):
        for key, tensor in a[component].state_dict().items():
            assert torch.equal(tensor, b[component].state_dict()[key])
    assert a[3] == b[3]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("single_source")
    c = config()
    c["training"] = tiny_training()
    c["scenario_sampling"]["development"]["count"] = 1
    c["scenario_sampling"]["heldout"]["count"] = 1
    c["optimization"]["additional_seed_scenarios"] = ["heldout_001"]
    c["reference"]["candidates"] = [dict(nx=n, ny=n, dt=.004/f**2, steps=25*f**2)
                                     for n, f in ((11, 1), (21, 2), (41, 4))]
    c["reference"]["selected_grid"] = 41
    c["evaluation_times"] = [.03, .1]
    c["measurement_times"] = dict(start=0., stop=.1, count=7)
    c["diagnostics"].update(interior_count=12, initial_count=8, boundary_per_edge=4, prediction_batch=256)
    c["identifiability"]["grid_count"] = 3
    inference = yaml.safe_load((ROOT/"configs/reference_inference.yaml").read_text())
    inference["solver"] = dict(nx=17, ny=17, dt=.004, steps=25)
    inference_path = temporary/"inference.yaml"
    inference_path.write_text(yaml.safe_dump(inference))
    c["reference"]["inference_config"] = str(inference_path)
    output = temporary/"benchmark"
    _, plan = freeze_protocol(c, output)
    for scenario in plan:
        generate_scenario(output, scenario)
    freeze_manifest(output)
    return output, c, plan


def test_geometry_and_zero_noise_are_bitwise_identical_across_actual_references(generated):
    root, c, plan = generated
    geometry = np.asarray(json.loads((root/"sensor_geometry.json").read_text()))
    for scenario in plan:
        directory = root/"datasets"/scenario["scenario_id"]
        inputs = load_model_inputs(directory)
        with np.load(directory/"truth/grid_41/checkpoint.npz") as truth:
            np.testing.assert_array_equal(inputs["measurements"], truth["measurements"])
            np.testing.assert_array_equal(inputs["positions"], geometry)
        manifest = json.loads((directory/"generation.json").read_text())
        assert manifest["sensor_layout_hash"] == sha256(root/"sensor_geometry.json")
        assert manifest["noise_std"] == 0


def test_protocol_freeze_cannot_be_overwritten_and_manifest_is_stable(generated):
    root, c, _ = generated
    before = (root/"scenario_manifest.csv").read_bytes()
    freeze_manifest(root)
    assert (root/"scenario_manifest.csv").read_bytes() == before
    assert load_frozen(root)[0] == c
    with pytest.raises(FileExistsError):
        freeze_protocol(c, root)


def test_plan_or_sensor_mutation_is_detected_before_running(tmp_path):
    c = config()
    root = tmp_path/"frozen"
    freeze_protocol(c, root)
    plan = json.loads((root/"scenario_plan.json").read_text())
    plan[0]["source"]["x_s"] += .01
    (root/"scenario_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash changed"):
        load_frozen(root)


def test_noise_nonzero_is_rejected_before_freezing(tmp_path):
    with pytest.raises(ValueError, match="zero measurement noise"):
        freeze_protocol(config() | {"noise_std": .001}, tmp_path/"invalid")


def test_truth_loading_refuses_to_start_before_training(tmp_path, monkeypatch):
    original = Path.read_text
    opened = []
    def guard(path, *args, **kwargs):
        opened.append(path.name)
        assert path.name == "attempt.json"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guard)
    with pytest.raises(FileNotFoundError):
        load_evaluation_truth(tmp_path, "heldout_001", tmp_path)
    assert opened == ["attempt.json"]


def test_end_to_end_truth_only_after_fit_and_training_unchanged_by_truth(generated, tmp_path, monkeypatch):
    root, c, _ = generated
    directory = root/"datasets/development_001"
    manifest = json.loads((directory/"generation.json").read_text())
    recipe = dict(training=c["training"], sigma=.08, optimization_seed=42)
    hashes = {"observations.npz": manifest["observations_hash"], "physics.json": manifest["physics_hash"]}
    original_load, original_read = np.load, Path.read_text
    def deny_truth_load(path, *args, **kwargs):
        assert "truth" not in Path(path).parts
        return original_load(path, *args, **kwargs)
    def deny_truth_read(path, *args, **kwargs):
        assert "truth" not in path.parts and path.name not in ("scenario_plan.json", "generation.json")
        return original_read(path, *args, **kwargs)
    script = runner()
    monkeypatch.setattr(np, "load", deny_truth_load)
    monkeypatch.setattr(Path, "read_text", deny_truth_read)
    first = script.train_attempt(directory, tmp_path/"first", recipe, hashes)
    assert first["computational_success"]
    second = script.train_attempt(directory, tmp_path/"second", recipe, hashes)
    assert second["computational_success"]
    a = torch.load(tmp_path/"first/model.pt", weights_only=True)
    b = torch.load(tmp_path/"second/model.pt", weights_only=True)
    for group in ("model_state_dict", "source_state_dict"):
        for name in a[group]:
            assert torch.equal(a[group][name], b[group][name])


def test_failed_fit_is_retained_with_truth_and_null_metrics(generated, monkeypatch):
    root, c, _ = generated
    script = runner()
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate nonfinite gradient")
    monkeypatch.setattr(script, "fit_sparse_source", fail)
    row = script.train_and_evaluate(root, "heldout_001", 42)
    assert not row["computational_success"] and not row["recovery_success"]
    assert row["localization_error"] is None
    assert "deliberate nonfinite" in row["failure"]
    assert row["x_true"] > 0 and row["Q_true"] > 0
    attempt = root/"runs/heldout_001/seed_42/attempt.json"
    before = attempt.read_bytes()
    # Every remaining attempt is made independently; the failed seed is skipped.
    assert (attempt.parent/"metrics.json").exists()
    assert json.loads(before)["training_terminated"]


def test_successful_fit_evaluates_source_and_dense_metrics(generated):
    root, c, _ = generated
    row = runner().train_and_evaluate(root, "development_001", 42)
    assert row["computational_success"] and row["evaluation_success"]
    assert row["relative_l2"] >= 0 and row["pde_rmse"] >= 0
    assert 0 <= row["negative_fraction"] <= 1
    assert (root/"runs/development_001/seed_42/snapshot_metrics.json").exists()


def test_exact_source_and_concentration_metrics():
    result = parameter_metrics(dict(x_s=.3, y_s=.4, Q=1.5), dict(x_s=0., y_s=0., Q=2.))
    assert result == pytest.approx(dict(localization_error=.5, relative_strength_error=.25))
    result = concentration_metrics(np.array([[-1., 4.]]), np.array([[1., 2.]]))
    assert result == pytest.approx(dict(relative_l2=np.sqrt(8/5), rmse=2., mae=2.))


def test_computational_and_scientific_success_are_distinct():
    rule = config()["recovery"]
    assert not recovery_decision(True, dict(localization_error=.3, relative_strength_error=.01), rule)
    assert not recovery_decision(True, dict(localization_error=.01, relative_strength_error=.9), rule)
    assert not recovery_decision(False, dict(localization_error=0., relative_strength_error=0.), rule)
    assert recovery_decision(True, dict(localization_error=.04, relative_strength_error=.1), rule)


def test_independent_diagnostics_on_exact_affine_solution():
    class Field(torch.nn.Module):
        def forward(self, points):
            return points[:, 2:3]
    class Source(torch.nn.Module):
        def forward(self, x, y):
            return torch.ones_like(x)
    settings = dict(seed=99, interior_count=16, initial_count=8, boundary_per_edge=4, prediction_batch=8)
    points = diagnostic_points(settings, .8)
    again = diagnostic_points(settings, .8)
    assert all(torch.equal(a, b) for a, b in zip(points, again))
    result = physical_diagnostics(Field(), Source(), dict(u=.35, v=-.15, D=.005), .8, settings)
    # C=t gives Ct-S=1-1=0 and IC=0, but violates zero BC by precisely t.
    assert result["pde_rmse"] == 0 and result["ic_rmse"] == 0
    assert result["bc_rmse"] == pytest.approx(points[2][:, 2].double().square().mean().sqrt().item())
    assert result["bc_mae"] > 0


def test_summary_sample_std_quantiles_missing_and_wilson_are_correct():
    result = distribution([1., 2., 3., None], 4)
    assert result == pytest.approx(dict(count=3, missing=1, mean=2, median=2, std=1,
                                       q25=1.5, q75=2.5, q90=2.8, max=3))
    interval = wilson_interval(0, 30, .95)
    assert interval[0] == pytest.approx(0, abs=1e-15)
    assert interval[1] == pytest.approx(.113513393, abs=1e-8)


def test_failed_runs_in_denominator_restarts_not_pooled():
    base = dict(scenario_id="heldout_001", split="heldout", computational_success=True,
        evaluation_success=True, recovery_success=True, reference_target_met=True,
        localization_error=.01, relative_strength_error=.1, relative_l2=.2)
    rows = [base | dict(optimization_seed=42), base | dict(optimization_seed=314),
            base | dict(scenario_id="heldout_002", optimization_seed=42, computational_success=False,
                        evaluation_success=False, recovery_success=False, localization_error=None)]
    result = summarize(rows, config())
    assert result["heldout"]["total"] == 2
    assert result["heldout"]["recovery_rate"] == .5
    assert result["heldout"]["metrics"]["localization_error"]["missing"] == 1
    assert result["all_runs"]["total"] == 3


def test_raw_serialization_deterministic_excluding_runtime(tmp_path):
    rows = [dict(scenario_id="heldout_001", localization_error=.0123,
                 computational_success=True, recovery_success=False, missing=None)]
    for name in ("a", "b"):
        write_csv(tmp_path/f"{name}.csv", rows)
    assert (tmp_path/"a.csv").read_bytes() == (tmp_path/"b.csv").read_bytes()
    assert object_hash(rows) == object_hash(json.loads(json.dumps(rows)))


def test_complete_report_and_controlled_landscape_keep_failed_primary(generated):
    root, c, _ = generated
    script = runner()
    # Do not depend on another test having created a particular attempt.
    for scenario_id, seed in script.planned_runs(c, scenario_plan(c)):
        if not (root/"runs"/scenario_id/f"seed_{seed}"/"attempt.json").exists():
            script.train_and_evaluate(root, scenario_id, seed)
    landscape = script.run_landscape(root)
    assert not landscape["probability_map"]
    assert landscape["true_location_mse"] >= 0
    assert (root/"source_location_loss_landscape.png").exists()
    result = script.report(root)
    assert result["heldout"]["total"] == 1
    assert result["all_runs"]["total"] == 4
    assert len((root/"raw_results.csv").read_text().splitlines()) == 5
    for name in ("true_vs_predicted_sources", "localization_error_distribution",
                 "source_strength_recovery", "physical_diagnostics"):
        assert (root/f"{name}.png").exists() and (root/f"{name}.pdf").exists()
    before = (root/"raw_results.csv").read_bytes()
    with pytest.raises(FileExistsError):
        script.report(root)
    assert (root/"raw_results.csv").read_bytes() == before


def test_resume_never_retries_recorded_failures(generated, monkeypatch):
    root, c, _ = generated
    script = runner()
    # Explicit resume with no pending tasks must not call any optimizer again.
    # Create a retained failure if run in isolation instead of after other tests.
    for scenario_id, seed in script.planned_runs(c, scenario_plan(c)):
        output = root/"runs"/scenario_id/f"seed_{seed}"
        if not (output/"attempt.json").exists():
            script.train_and_evaluate(root, scenario_id, seed)
    before = {str(p): p.read_bytes() for p in (root/"runs").rglob("attempt.json")}
    def forbidden(*args, **kwargs):
        pytest.fail("A terminal attempt must not be rerun")
    monkeypatch.setattr(script, "train_and_evaluate", forbidden)
    script.run_training(root, resume=True)
    assert before == {str(p): p.read_bytes() for p in (root/"runs").rglob("attempt.json")}
