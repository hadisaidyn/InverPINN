"""Frozen scenario generation and reference/input separation for revision 3."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import yaml

from inverpinn.data.synthetic_reference import ROOT, UNITS, provenance, run_refinement, sha256, write_json


def object_hash(value):
    """Hash canonical JSON (no timestamps); used for immutable scientific plans."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    """Preserve supplied row/column order and explicit missing values in CSV."""
    if not rows:
        raise ValueError("Cannot serialize an empty results table.")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def scenario_plan(config):
    """Independent uniform xs,ys,Q draws from split-specific SeedSequences.

    Sampling has no dependency on observations, initialization or fit outcomes.
    A plan is frozen before simulation; re-running the function reproduces it.
    """
    settings = config["scenario_sampling"]
    a, b = settings["position_range"]
    q0, q1 = settings["strength_range"]
    if not (0 < a < b < 1 and 0 < q0 < q1):
        raise ValueError("Require interior source ranges and positive ordered Q range.")
    result = []
    for split in ("development", "heldout"):
        spec = settings[split]
        children = np.random.SeedSequence(spec["seed"]).spawn(spec["count"])
        for index, child in enumerate(children, 1):
            seed = int(child.generate_state(1, dtype=np.uint64)[0]) % (2**63-1)
            random = np.random.default_rng(seed)
            source = dict(x_s=float(random.uniform(a, b)), y_s=float(random.uniform(a, b)),
                          Q=float(random.uniform(q0, q1)), sigma=config["source_family"]["sigma"])
            result.append(dict(scenario_id=f"{split}_{index:03d}", split=split,
                               generation_seed=seed, source=source, source_hash=object_hash(source)))
    if len({r["scenario_id"] for r in result}) != len(result) or len({r["source_hash"] for r in result}) != len(result):
        raise ValueError("Scenario IDs/source parameters must not overlap.")
    return result


def freeze_protocol(config, output):
    """Create a non-overwriting protocol/code snapshot before any references."""
    output = Path(output)
    if config["noise_std"] != 0:
        raise ValueError("Revision 3 permits exactly zero measurement noise.")
    if config["source_family"]["kind"] != "gaussian_peak":
        raise ValueError("Only one known-width Gaussian peak source is supported.")
    sensors_path = ROOT/config["sensors_file"]
    sensors = np.asarray(json.loads(sensors_path.read_text()), dtype=float)
    if sensors.shape != (config["sensor_count"], 2) or not np.isfinite(sensors).all() or not ((sensors >= 0) & (sensors <= 1)).all():
        raise ValueError("Invalid fixed sensor geometry.")
    plan = scenario_plan(config)
    heldout_ids = {s["scenario_id"] for s in plan if s["split"] == "heldout"}
    if not set(config["optimization"]["additional_seed_scenarios"]) <= heldout_ids:
        raise ValueError("Additional-seed subset must name held-out scenarios.")
    output.mkdir(parents=True, exist_ok=False)
    (output/"protocol.yaml").write_text(yaml.safe_dump(config))
    shutil.copyfile(sensors_path, output/"sensor_geometry.json")
    shutil.copyfile(ROOT/"docs/single_source_benchmark_protocol.md", output/"protocol.md")
    write_json(output/"scenario_plan.json", plan)
    run_provenance = provenance(output)
    # Include ALL actual executable project modules (including dirty new code),
    # not only the solver modules archived by the revision-2 helper.
    code = sorted([*ROOT.glob("src/inverpinn/**/*.py"), *ROOT.glob("scripts/*.py")])
    with zipfile.ZipFile(output/"benchmark_code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(path.relative_to(ROOT)))
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=object_hash(config), plan_hash=object_hash(plan),
        sensor_layout_hash=sha256(output/"sensor_geometry.json"),
        protocol_document_hash=sha256(output/"protocol.md"),
        reference_solver_commit=run_provenance["git_commit"],
        reference_solver_hash=sha256(ROOT/"src/inverpinn/physics/finite_difference.py"),
        code_hashes={str(p.relative_to(ROOT)): sha256(p) for p in code},
        inference_config_hash=sha256(ROOT/config["reference"]["inference_config"]))
    write_json(output/"protocol_lock.json", frozen)
    return config, plan


def load_frozen(output, verify_code=True):
    """Refuse changed plans/settings/code instead of silently changing a run."""
    output = Path(output)
    config = yaml.safe_load((output/"protocol.yaml").read_text())
    plan = json.loads((output/"scenario_plan.json").read_text())
    lock = json.loads((output/"protocol_lock.json").read_text())
    checks = [object_hash(config) == lock["protocol_hash"], object_hash(plan) == lock["plan_hash"],
              sha256(output/"sensor_geometry.json") == lock["sensor_layout_hash"],
              sha256(output/"protocol.md") == lock["protocol_document_hash"],
              sha256(ROOT/config["reference"]["inference_config"]) == lock["inference_config_hash"]]
    if verify_code:
        checks.extend(sha256(ROOT/name) == digest for name, digest in lock["code_hashes"].items())
    if not all(checks):
        raise ValueError("Frozen protocol, scenario plan, sensor geometry or code hash changed.")
    return config, plan, lock


def generate_scenario(root, scenario):
    """Run three-grid qualification; deliver exactly clean sparse observations.

    The finest preregistered grid is always used, even if a coarser grid would
    qualify or the 1% conditional estimate is missed. Such misses are flagged.
    """
    root = Path(root)
    config, _, lock = load_frozen(root)
    output = root/"datasets"/scenario["scenario_id"]
    output.mkdir(parents=True, exist_ok=False)
    reference = config["reference"]
    settings = dict(seed=scenario["generation_seed"], torch_threads=config["runtime"]["torch_threads"],
        source=scenario["source"], physics=config["physics"], domain=config["domain"],
        evaluation_times=config["evaluation_times"], measurement_times=config["measurement_times"],
        sensors={"positions": json.loads((root/"sensor_geometry.json").read_text())},
        candidates=reference["candidates"], target_relative_L2=reference["target_relative_L2"],
        inference_config=reference["inference_config"])
    accuracy, _ = run_refinement(settings, output/"truth")
    grid = reference["selected_grid"]
    checkpoint = output/"truth"/f"grid_{grid}"/"checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as truth:
        inputs = output/"inputs"
        inputs.mkdir()
        # No random noise operation: IEEE bitwise equality, not merely small RMS.
        np.savez_compressed(inputs/"observations.npz", positions=truth["sensor_positions"],
                            times=truth["measurement_times"], measurements=truth["measurements"])
        if not np.array_equal(truth["sensor_positions"], np.asarray(settings["sensors"]["positions"])):
            raise ValueError("Reference changed the fixed geometry.")
    write_json(inputs/"physics.json", dict(transport=config["physics"], domain=config["domain"],
        initial_condition="C=0 everywhere", boundary_condition="C=0 on all four edges",
        units={k: v for k, v in UNITS.items() if k not in ("Q", "sigma")}))
    write_json(output/"truth"/"source.json", scenario["source"])
    row = dict(scenario_id=scenario["scenario_id"], split=scenario["split"],
        generation_seed=scenario["generation_seed"], source_hash=scenario["source_hash"],
        reference_data_hash=sha256(checkpoint), reference_solver_commit=lock["reference_solver_commit"],
        reference_solver_hash=lock["reference_solver_hash"], sensor_layout_hash=lock["sensor_layout_hash"],
        observations_hash=sha256(inputs/"observations.npz"), physics_hash=sha256(inputs/"physics.json"),
        reference_path=str(checkpoint.relative_to(root)), noise_std=0.0,
        max_reference_error_estimate=accuracy["per_grid"][str(grid)]["max_assessed_Richardson_relative_L2_estimate"],
        reference_target_met=accuracy["per_grid"][str(grid)]["meets_estimated_target"])
    write_json(output/"generation.json", row)
    return row


def freeze_manifest(root):
    """Lock reference/input hashes in plan order, before held-out training."""
    root = Path(root)
    _, plan, lock = load_frozen(root)
    rows = [json.loads((root/"datasets"/s["scenario_id"]/"generation.json").read_text()) for s in plan]
    if (root/"manifest_lock.json").exists():
        if object_hash(rows) != json.loads((root/"manifest_lock.json").read_text())["manifest_hash"]:
            raise ValueError("Frozen manifest changed.")
        return rows
    write_csv(root/"scenario_manifest.csv", rows)
    write_json(root/"manifest_lock.json", dict(manifest_hash=object_hash(rows),
        manifest_csv_hash=sha256(root/"scenario_manifest.csv"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), protocol_hash=lock["protocol_hash"]))
    return rows
