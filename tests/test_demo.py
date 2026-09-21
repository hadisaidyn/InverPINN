"""Presentation-only checks: never train, infer, or solve the scientific PDE."""

import ast
import copy
import importlib
import importlib.abc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.evidence import (
    DATA,
    EVIDENCE,
    FILTERS,
    IDS,
    METRICS,
    csv_rows,
    filter_scenarios,
    frozen_metadata,
    load_bundle,
    load_fields,
    recovered,
    sha,
)
from demo.plots import FIELD_MODES, field_plot, location_plot, q_plot
from scripts.build_demo_data import build, save_npz


@pytest.fixture(scope="module")
def metadata():
    return load_bundle()[0]


def test_only_exact_final_ids_and_canonical_summary(metadata):
    assert [s["scenario_id"] for s in metadata["scenarios"]] == list(IDS)
    assert metadata == frozen_metadata()
    assert metadata["required"] == 27
    canonical = json.loads((ROOT / "paper/generated/summary.json").read_text())
    for method, count in [("J1", 23), ("classical", 30)]:
        assert metadata["summary"][method] == canonical["methods"][method]
        assert (
            sum(s["methods"][method]["recovery_success"] for s in metadata["scenarios"])
            == count
        )
    assert canonical["confirmation_passed"] is False
    assert metadata["summary"]["J1"]["Q_under"] == 28
    assert metadata["summary"]["J1"]["Q_over"] == 2


@pytest.mark.parametrize("method", ["J1", "classical"])
def test_all_displayed_source_values_and_metrics_match_csv(metadata, method):
    raw = csv_rows(EVIDENCE / f"final/{method}_raw.csv")
    for scenario in metadata["scenarios"]:
        source = raw[scenario["scenario_id"]]
        for key in ("x_true", "y_true", "Q_true"):
            assert scenario["truth"][key] == float(source[key])
        for key in ("x_pred", "y_pred", "Q_pred", *METRICS, "rmse", "mae"):
            assert scenario["methods"][method][key] == float(source[key])
        assert scenario["observations_hash"] == source["observations_hash"]
        assert scenario["reference_data_hash"] == source["reference_data_hash"]


def test_frozen_recovery_boundary_and_filters(metadata):
    criterion = metadata["criterion"]
    assert recovered(0.08, 0.2, criterion)
    assert not recovered(np.nextafter(0.08, 1), 0.2, criterion)
    assert not recovered(0.08, np.nextafter(0.2, 1), criterion)
    assert [
        len(filter_scenarios(metadata["scenarios"], label)) for label in FILTERS
    ] == [30, 23, 7, 30]
    assert len(metadata["scenarios"]) == 30  # Filtering does not delete global cases.
    assert metadata["paper_cases"] == {"success": IDS[2], "failure": IDS[29]}
    assert metadata["default_scenario"] == IDS[0]


@pytest.mark.parametrize("sid", IDS)
def test_finite_correct_fields_and_stored_plot_coordinates(metadata, sid):
    fields = load_fields(sid)
    scenario = next(s for s in metadata["scenarios"] if s["scenario_id"] == sid)
    scales = []
    for mode in FIELD_MODES:
        plot = field_plot(scenario, fields, metadata["sensors"], mode, 3)
        assert np.isfinite(plot.data[0].z).all()
        if mode != FIELD_MODES[2]:
            scales.append(plot.data[0].zmax)
        assert len(plot.data[1].x) == 20
        for trace in plot.data:
            assert np.isfinite(trace.x).all() and np.isfinite(trace.y).all()
        for trace, method in [(plot.data[2], "classical"), (plot.data[3], "J1")]:
            assert trace.x[0] == scenario["methods"][method]["x_pred"]
            assert trace.y[0] == scenario["methods"][method]["y_pred"]
        assert plot.data[4].x[0] == scenario["truth"]["x_true"]
    assert len(set(scales)) == 1
    np.testing.assert_array_equal(
        field_plot(scenario, fields, metadata["sensors"], FIELD_MODES[0], 0).data[0].z,
        fields["reference"][0],
    )


def test_truth_does_not_replace_displayed_estimates(metadata):
    scenario = copy.deepcopy(metadata["scenarios"][0])
    fields = load_fields(IDS[0])
    before = field_plot(scenario, fields, metadata["sensors"], FIELD_MODES[1], 3)
    scenario["truth"] = {"x_true": 0.99, "y_true": 0.01, "Q_true": 99.0}
    after = field_plot(scenario, fields, metadata["sensors"], FIELD_MODES[1], 3)
    for index in (0, 2, 3):
        assert before.data[index].to_json() == after.data[index].to_json()
    assert before.data[4].x != after.data[4].x
    q = q_plot([scenario])
    assert q.data[1].y[0] == scenario["methods"]["J1"]["Q_pred"]


def test_global_plots_keep_every_case(metadata):
    q = q_plot(metadata["scenarios"])
    assert len(q.data[1].x) == len(q.data[2].x) == 30
    assert list(q.data[0].x) == list(q.data[0].y)
    for toggle, expected in [(False, 2), (True, 3)]:
        points = [
            t
            for t in location_plot(metadata["scenarios"], toggle).data
            if t.mode == "markers"
        ]
        assert len(points) == expected
        assert all(
            len(t.x) == 30 and np.isfinite(t.x).all() and np.isfinite(t.y).all()
            for t in points
        )


def test_changed_bundle_stops_loudly(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(DATA, target)
    (target / "final_metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="bundle mismatch"):
        load_bundle(target)


def test_unknown_scenario_rejected_and_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown final"):
        load_fields("development_v3_001")
    bad = tmp_path / "bad.csv"
    bad.write_text("scenario_id\n" + (IDS[0] + "\n") * 30)
    with pytest.raises(ValueError, match="30 final scenarios"):
        csv_rows(bad)


def test_deterministic_display_serialization(tmp_path):
    target = tmp_path / "display.npz"
    save_npz(target, load_fields(IDS[0]))
    assert sha(target) == sha(DATA / f"fields/{IDS[0]}.npz")
    with pytest.raises(FileExistsError):
        save_npz(target, load_fields(IDS[0]))


def test_full_bundle_regeneration_from_preserved_archive(tmp_path):
    inventory = ROOT / "artifacts/finalization_audit/protected_before.json"
    archive = ROOT / "results/final_confirmatory"
    if not inventory.is_file() or not archive.is_dir():
        pytest.skip(
            "Full regeneration requires the preserved research archive, not shipped in a clean clone"
        )
    target = tmp_path / "regenerated"
    rebuilt = build(target)
    assert rebuilt == json.loads((DATA / "manifest.json").read_text())
    assert load_bundle(target)[0] == load_bundle()[0]
    with pytest.raises(FileExistsError):
        build(target)


@pytest.mark.parametrize(
    "path",
    [
        "src/demo-output",
        "configs/demo-output",
        "results/demo-output",
        "paper/evidence/demo-output",
    ],
)
def test_builder_cannot_write_scientific_paths(path):
    with pytest.raises(ValueError, match="Output must"):
        build(ROOT / path)


def test_no_scientific_imports_or_execution_interfaces():
    files = [*sorted((ROOT / "demo").glob("*.py")), ROOT / "scripts/build_demo_data.py"]
    for path in files:
        tree = ast.parse(path.read_text())
        imports = [
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ]
        imports += [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        assert not any(
            n.split(".")[0] in ("torch", "inverpinn", "scipy", "subprocess")
            for n in imports
        )
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("eval", "exec")
            for n in ast.walk(tree)
        )


def test_app_imports_and_interactions_without_scientific_modules(metadata):
    # Runtime guard supplements static imports; the actual app executes here.
    class RejectScience(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("inverpinn", "torch", "scipy"):
                raise AssertionError(f"Scientific runtime imported: {fullname}")

    guard = RejectScience()
    sys.meta_path.insert(0, guard)
    try:
        app = importlib.import_module("demo.app")
        assert callable(app.main)
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "demo/app.py")).run(timeout=30)
        assert not at.exception
        assert at.selectbox(key="scenario").value == IDS[0]
        assert [m.value for m in at.metric] == [
            "23 / 30 recovered",
            "27 / 30",
            "30 / 30 recovered",
        ]
        at.selectbox(key="failure_filter").set_value("Failed by J1").run()
        assert len(at.selectbox(key="scenario").options) == 7
        at.selectbox(key="scenario").set_value(IDS[29]).run()
        worst = metadata["scenarios"][29]["methods"]["J1"]
        assert any(
            f"{100*worst['relative_strength_error']:.3f}%" in m.value
            for m in at.markdown
        )
        for mode in FIELD_MODES:
            at.selectbox[2].set_value(mode).run()
            assert not at.exception
        at.select_slider[0].set_value(0.1).run()
        at.checkbox[0].check().run()
        assert not at.exception
        at.button[0].click().run()
        assert at.selectbox(key="scenario").value == IDS[2]
        assert at.selectbox(key="failure_filter").value == FILTERS[0]
        at.selectbox(key="scenario").set_value(IDS[15]).run()
        assert any("reference flag" in w.value for w in at.warning)
        assert any("FAILED" in e.value for e in at.error)
    finally:
        sys.meta_path.remove(guard)


def test_readme_commands_exist_and_frozen_science_unchanged():
    for path in ("README.md", "demo/README.md"):
        text = (ROOT / path).read_text()
        assert "pip install -r demo/requirements.txt" in text
        assert "streamlit run demo/app.py" in text
    import subprocess

    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "2badcf1639faeb136a29567552109b81aade8ade",
            "--",
            "src",
            "configs",
            "data",
            "results",
            "paper",
            "presentation",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
