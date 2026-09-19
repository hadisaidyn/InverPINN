"""Orchestration tests run manufactured data and tiny training budgets, not a paper."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def runner():
    spec = importlib.util.spec_from_file_location("paper_runner", Path(__file__).parents[1]/"scripts/reproduce_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke():
    return yaml.safe_load(Path("configs/paper_smoke.yaml").read_text())


def config_file(tmp_path, config):
    path = tmp_path/"paper.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_full_manifest_dependencies_and_preserved_scientific_budgets(tmp_path):
    mod = runner()
    config = yaml.safe_load(Path("configs/paper.yaml").read_text())
    stages, snapshots = mod.prepare(config, tmp_path/"output")
    by_id = {s["id"]:s for s in stages}
    assert by_id["inverse_pinn"]["settings"]["steps"] == 6000
    assert by_id["sensor_sweep"]["settings"]["seeds"] == [0,1,2,3,4]
    assert by_id["sensor_sweep"]["settings"]["sensor_counts"] == [5,10,20,50]
    assert by_id["noise_sweep"]["settings"]["noise_percentages"] == [0,2,5,10,20]
    assert by_id["real_case_study"]["status"] == "excluded"
    assert all(p.is_relative_to(tmp_path/"output"/"configs") for p in snapshots)
    assert str(tmp_path/"output"/"experiments"/"dataset") in by_id["data_baseline"]["settings"]["dataset"]


@pytest.mark.parametrize("problem", ["duplicate", "forward", "escape", "unknown", "exclusion", "dpi", "seed"])
def test_invalid_manifests_fail_before_output_creation(tmp_path, problem):
    config = smoke()
    if problem == "duplicate": config["stages"][1]["id"] = "dataset"
    if problem == "forward": config["stages"][1]["overrides"]["dataset"] = "${absent}/dataset.npz"
    if problem == "escape": config["stages"][0]["required"] = ["../outside"]
    if problem == "unknown": config["stages"][0]["kind"] = "shell"
    if problem == "exclusion": config["stages"][0]["enabled"] = False
    if problem == "dpi": config["publication"]["dpi"] = 72
    if problem == "seed": config["seed"] = -1
    path = config_file(tmp_path, config)
    with pytest.raises(ValueError): runner().reproduce(path, tmp_path/"result")
    assert not (tmp_path/"result").exists()


def test_git_unavailable_is_explicit_and_strict_mode_blocks(tmp_path, monkeypatch):
    mod = runner()
    assert mod.git_info(tmp_path)["commit"] is None
    monkeypatch.setattr(mod, "git_info", lambda _:dict(commit=None, status="unavailable"))
    config = smoke() | dict(require_git_commit=True)
    with pytest.raises(ValueError, match="Git commit"):
        mod.reproduce(config_file(tmp_path, config), tmp_path/"result")
    assert not (tmp_path/"result").exists()


def test_existing_output_preserved(tmp_path):
    output = tmp_path/"result"
    output.mkdir()
    (output/"precious.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        runner().reproduce(config_file(tmp_path, smoke()), output)
    assert [p.name for p in output.iterdir()] == ["precious.txt"]
    assert (output/"precious.txt").read_text() == "keep"


def test_smoke_end_to_end_twice_numeric_identity_and_provenance(tmp_path):
    mod = runner()
    path = config_file(tmp_path, smoke())
    outputs = [mod.reproduce(path, tmp_path/name) for name in ("first", "second")]
    for output in outputs:
        assert json.loads((output/"run.json").read_text())["status"] == "complete"
        environment = json.loads((output/"environment.json").read_text())
        assert "torch" in {k.lower() for k in environment["packages"]}
        assert environment["process_settings"]["PYTHONHASHSEED"] == "42"
        assert "commit" in json.loads((output/"git.json").read_text())
        assert list((output/"paper/figures").glob("*.pdf"))
        assert (output/"paper/tables/interpolation__metrics.csv").is_file()
        for config in (output/"experiments").rglob("config.yaml"):
            provenance = json.loads((config.parent/"paper_provenance.json").read_text())
            assert provenance["config_sha256"] == mod.sha256(config)
        for row in json.loads((output/"artifacts.json").read_text()):
            assert row["sha256"] == mod.sha256(output/row["path"])
        assert (output/"paper/figures/dataset__sensor_positions.pdf").read_bytes().startswith(b"%PDF")
        from PIL import Image
        with Image.open(output/"paper/figures/dataset__sensor_positions.png") as im:
            assert im.info["dpi"][0] == pytest.approx(300, abs=.1)
    for stage in ("data_baseline", "inverse_pinn", "inverse_evaluation"):
        a,b = [json.loads((p/"experiments"/stage/"metrics.json").read_text()) for p in outputs]
        # Evaluator includes input/output paths; compare scalar numeric metrics.
        for key,value in a.items():
            if isinstance(value, (int,float)):
                assert b[key] == value
    for stage in ("dataset", "inverse_pinn"):
        name = "dataset.npz" if stage == "dataset" else "predictions.npz"
        with np.load(outputs[0]/"experiments"/stage/name) as a, np.load(outputs[1]/"experiments"/stage/name) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_failed_stage_stops_dependents_and_retains_evidence(tmp_path):
    config = smoke()
    config["stages"][0]["required"].append("nonexistent.pdf")
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="Required artifact"):
        runner().reproduce(config_file(tmp_path, config), output)
    state = json.loads((output/"run.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"][0]["status"] == "failed"
    assert state["stages"][1]["status"] == "pending"
    assert (output/"logs/dataset.log").exists()
    assert (output/"configs/dataset.yaml").exists()
    assert not (output/"paper").exists()
