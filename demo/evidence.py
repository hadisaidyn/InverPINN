"""Read and verify saved results, never infer sources or evaluate a model.

Truth is joined to already stored estimates solely for visualization. Relative
errors and recovery labels are copied from frozen tables and cross-checked,
not recalculated from display-resolution fields. All units are synthetic.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper/evidence"
DATA = ROOT / "demo/data"
IDS = tuple(f"final_blind_v1_{n:03d}" for n in range(1, 31))
METRICS = ("localization_error", "relative_strength_error", "relative_l2")
METHODS = ("J1", "classical")
FILTERS = (
    "All final scenarios",
    "Recovered by J1",
    "Failed by J1",
    "Recovered by classical",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(
        len(rows) == 30 and {r["scenario_id"] for r in rows} == set(IDS),
        f"Not exactly the 30 final scenarios: {path.name}",
    )
    return {r["scenario_id"]: r for r in rows}


def recovered(localization, q_error, criterion):
    """Frozen secondary joint endpoint; both inequalities must hold."""
    return (
        localization <= criterion["secondary_localization_max"]
        and q_error <= criterion["secondary_relative_strength_max"]
    )


def frozen_metadata():
    """Validate the sealed inputs and return a lossless JSON-friendly subset."""
    manifest = read_json(EVIDENCE / "manifest.json")
    names = [
        "final/J1_raw.csv",
        "final/classical_raw.csv",
        "final/manifest.csv",
        "final/observability.csv",
        "final/sensor_geometry.json",
        "final/decision.json",
        "configs/final_confirmatory.yaml",
    ]
    for name in names:
        require(
            sha(EVIDENCE / name) == manifest["files"][name]["sha256"],
            f"Changed frozen input: {name}",
        )
    canonical = ROOT / "paper/generated"
    require(
        sha(canonical / "summary.json")
        == read_json(canonical / "artifact_hashes.json")["summary.json"],
        "Changed canonical summary",
    )
    summary = read_json(canonical / "summary.json")
    protocol = yaml.safe_load(
        (EVIDENCE / "configs/final_confirmatory.yaml").read_text()
    )
    criterion = protocol["recovery"]
    require(
        criterion["secondary_localization_max"] == 0.08
        and criterion["secondary_relative_strength_max"] == 0.2,
        "Recovery criterion changed",
    )
    require(
        protocol["advancement"]["min_recovered"] == 27, "Advancement threshold changed"
    )
    rows = {
        method: csv_rows(EVIDENCE / f"final/{method}_raw.csv") for method in METHODS
    }
    truths = csv_rows(EVIDENCE / "final/manifest.csv")
    observability = csv_rows(EVIDENCE / "final/observability.csv")
    sensors = read_json(EVIDENCE / "final/sensor_geometry.json")
    require(
        np.asarray(sensors).shape == (20, 2) and np.isfinite(sensors).all(),
        "Invalid sensor layout",
    )
    scenarios = []
    numeric = ("x_pred", "y_pred", "Q_pred", *METRICS, "rmse", "mae")
    for sid in IDS:
        truth = {key: float(truths[sid][key]) for key in ("x_true", "y_true", "Q_true")}
        methods = {}
        for method in METHODS:
            row = rows[method][sid]
            require(
                row["method"] == method and row["evaluation_success"] == "True",
                f"Invalid stored run: {sid}/{method}",
            )
            require(
                all(float(row[key]) == value for key, value in truth.items()),
                f"Truth mismatch: {sid}",
            )
            methods[method] = {key: float(row[key]) for key in numeric}
            require(
                np.isfinite(list(methods[method].values())).all(),
                f"Nonfinite metrics: {sid}",
            )
            require(
                all(0 <= methods[method][key] <= 1 for key in ("x_pred", "y_pred"))
                and methods[method]["Q_pred"] > 0,
                f"Invalid stored source: {sid}",
            )
            flag = row["recovery_success"] == "True"
            require(
                flag
                == recovered(
                    methods[method]["localization_error"],
                    methods[method]["relative_strength_error"],
                    criterion,
                ),
                f"Recovery label mismatch: {sid}/{method}",
            )
            methods[method]["recovery_success"] = flag
        for key in ("observations_hash", "reference_data_hash"):
            require(
                rows["J1"][sid][key] == rows["classical"][sid][key],
                f"Unpaired data: {sid}/{key}",
            )
        obs = observability[sid]
        scenarios.append(
            dict(
                scenario_id=sid,
                truth=truth,
                methods=methods,
                observations_hash=rows["J1"][sid]["observations_hash"],
                reference_data_hash=rows["J1"][sid]["reference_data_hash"],
                reference_target_met=rows["J1"][sid]["reference_target_met"] == "True",
                reference_error_estimate=float(
                    rows["J1"][sid]["reference_error_estimate"]
                ),
                observability={
                    k: float(obs[k])
                    for k in ("sensor_rms", "sigma_min", "jacobian_condition")
                },
                information_group=obs["information_group"],
            )
        )
    for method, count in [("J1", 23), ("classical", 30)]:
        require(
            sum(s["methods"][method]["recovery_success"] for s in scenarios) == count,
            "Recovery aggregate changed",
        )
        for key in METRICS:
            actual = median(s["methods"][method][key] for s in scenarios)
            require(
                abs(actual - summary["methods"][method]["metrics"][key]["median"])
                < 1e-14,
                "Canonical median mismatch",
            )
    require(
        sum(s["methods"]["J1"]["Q_pred"] < s["truth"]["Q_true"] for s in scenarios)
        == 28,
        "Q bias count changed",
    )
    require(
        read_json(EVIDENCE / "final/decision.json")["passed"] is False
        and summary["confirmation_passed"] is False,
        "Failed confirmation changed",
    )
    success = sorted(
        (s for s in scenarios if s["methods"]["J1"]["recovery_success"]),
        key=lambda s: (s["methods"]["J1"]["localization_error"], s["scenario_id"]),
    )
    worst = sorted(
        scenarios,
        key=lambda s: (-s["methods"]["J1"]["localization_error"], s["scenario_id"]),
    )[0]["scenario_id"]
    require(
        success[len(success) // 2]["scenario_id"]
        == manifest["selected_cases"]["success"]
        and worst == manifest["selected_cases"]["failure"],
        "Paper shortcut selection mismatch",
    )
    return dict(
        scenarios=scenarios,
        sensors=sensors,
        criterion=criterion,
        required=protocol["advancement"]["min_recovered"],
        summary={m: summary["methods"][m] for m in METHODS},
        paper_cases=manifest["selected_cases"],
        default_scenario=IDS[0],
        physics=protocol["physics"],
        sigma=protocol["source_family"]["sigma"],
        times=protocol["evaluation_times"],
        provenance={
            "scientific_commit": manifest["frozen_scientific_commit"],
            "input_hashes": {name: sha(EVIDENCE / name) for name in names},
            "canonical_summary_sha256": sha(canonical / "summary.json"),
        },
    )


def load_bundle(directory=DATA):
    """Check presentation data against committed scientific evidence at startup."""
    directory = Path(directory)
    bundle = read_json(directory / "manifest.json")
    require(
        set(bundle["files"])
        == {"final_metrics.json", *(f"fields/{sid}.npz" for sid in IDS)},
        "Incomplete or unexpected demo file inventory",
    )
    for name, record in bundle["files"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Unsafe bundle path")
        require(sha(path) == record["sha256"], f"Demo bundle mismatch: {name}")
    metadata = read_json(directory / "final_metrics.json")
    require(metadata == frozen_metadata(), "Demo metadata differs from frozen evidence")
    require(set(bundle["scenarios"]) == set(IDS), "Wrong field scenario set")
    return metadata, bundle


def load_fields(sid, directory=DATA):
    require(sid in IDS, "Unknown final scenario")
    directory = Path(directory)
    name = f"fields/{sid}.npz"
    manifest = read_json(directory / "manifest.json")
    require(
        sha(directory / name) == manifest["files"][name]["sha256"],
        f"Changed display field: {sid}",
    )
    with np.load(directory / name, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    require(
        set(values) == {"x", "y", "times", "reference", "J1", "classical"},
        "Invalid field keys",
    )
    require(
        all(np.isfinite(value).all() for value in values.values()),
        "Nonfinite display coordinates/fields",
    )
    require(
        all(values[k].shape == (4, 81, 81) for k in ("reference", "J1", "classical")),
        "Unexpected display grid",
    )
    # The archived solver's arithmetic and linspace can differ by one ULP.
    # Hash verification above still requires every archived display byte exactly.
    require(
        all(
            values[k].shape == (81,)
            and values[k][0] == 0
            and values[k][-1] == 1
            and np.allclose(
                values[k], np.linspace(0, 1, 81), rtol=0, atol=2 * np.finfo(float).eps
            )
            for k in ("x", "y")
        ),
        "Unexpected normalized coordinates",
    )
    require(
        np.array_equal(values["times"], [0.1, 0.2, 0.4, 0.8]), "Unexpected saved times"
    )
    return values


def filter_scenarios(scenarios, label):
    require(label in FILTERS, "Unknown scenario filter")
    if label == FILTERS[0]:
        return list(scenarios)
    method = "classical" if label == FILTERS[3] else "J1"
    status = label != FILTERS[2]
    return [s for s in scenarios if s["methods"][method]["recovery_success"] == status]
