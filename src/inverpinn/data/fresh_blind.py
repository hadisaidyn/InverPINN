"""Revision-5 prospective freeze, exclusion guards and immutable manifests.

Old sampling hashes are reconstructed from published deterministic rules only.
No old dataset or outcome is opened. Fresh truth is generated separately from
the input-only fitting manifest. Original solvers and trainers are never edited.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from inverpinn.data.single_source_benchmark import object_hash, scenario_plan, write_csv, load_frozen
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json, provenance

FROZEN_COMMIT = "9dc90c247b0a596b789cb0ee5f1c06f938db40c4"
REVISED_HASH = "3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e"
B0_HASH = "5f85e130900be9295bc48b3f152772a8f456b1a4b4b77f076258e1f6d8f30254"
FRESH_IDS = tuple(f"fresh_{i:03d}" for i in range(1, 31))


def fresh_path(root, identifier, relative=""):
    """Resolve an allowlisted fresh case; no old IDs, escapes or aliases."""
    if identifier not in FRESH_IDS:
        raise ValueError("Only fresh_001 through fresh_030 are allowed.")
    lexical = Path(root).resolve()/"datasets"/identifier
    case = lexical.resolve()
    path = (case/relative).resolve()
    if case != lexical or not path.is_relative_to(case):
        raise ValueError("Fresh path escapes its case or aliases another dataset.")
    return path


def check_open(root, path, fitting_case=None):
    """Deny previous result directories; during fitting deny fresh truth files.

    This is an accidental Python-path access guard, not an OS sandbox. Integer
    descriptors are ignored; subprocess reference generation uses only configs
    written by the guarded parent and the unchanged reference worker.
    """
    if isinstance(path, int):
        return
    path, root = Path(os.fsdecode(path)).resolve(), Path(root).resolve()
    results = ROOT/"results"
    if path.is_relative_to(results.resolve()) and not path.is_relative_to(root):
        raise PermissionError("Previous results are consumed and forbidden in revision 5.")
    if fitting_case is not None and path.is_relative_to(root):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "datasets":
            allowed = fresh_path(root, fitting_case, "inputs")
            if not path.is_relative_to(allowed):
                raise PermissionError("Fitting may open only this case's sparse inputs, never truth.")
        if relative.name in ("scenario_plan.json", "scenario_manifest.csv", "generation.json"):
            raise PermissionError("Truth-bearing plan/manifest is forbidden during fitting.")


def install_guard(root):
    """Mutable fitting phase for a process-lifetime open audit hook."""
    state = {"fitting_case": None}
    def audit(event, args):
        if event == "open":
            check_open(root, args[0], state["fitting_case"])
    sys.addaudithook(audit)
    return state


def fresh_plan(config, legacy):
    """Predetermined independent draws; reject collisions rather than resample."""
    spec = config["scenario_sampling"]
    old = scenario_plan(legacy)  # pure RNG/hash calculation, no data I/O
    old_seeds = {r["generation_seed"] for r in old}
    if spec["count"] != 30 or spec["seed"] in {legacy["scenario_sampling"][s]["seed"] for s in ("development", "heldout")}:
        raise ValueError("Require 30 scenarios and a new master seed.")
    children = np.random.SeedSequence(spec["seed"]).spawn(spec["count"])
    rows = []
    for identifier, child in zip(FRESH_IDS, children):
        seed = int(child.generate_state(1, dtype=np.uint64)[0]) % (2**63-1)
        rng = np.random.default_rng(seed)
        source = dict(x_s=float(rng.uniform(*spec["position_range"])),
                      y_s=float(rng.uniform(*spec["position_range"])),
                      Q=float(rng.uniform(*spec["strength_range"])), sigma=config["source_family"]["sigma"])
        rows.append(dict(scenario_id=identifier, split="fresh", generation_seed=seed,
                         source=source, source_hash=object_hash(source)))
    hashes, seeds = {r["source_hash"] for r in rows}, {r["generation_seed"] for r in rows}
    if len(hashes) != 30 or len(seeds) != 30 or hashes & {r["source_hash"] for r in old} or seeds & old_seeds:
        raise ValueError("Fresh/previous scenario collision: stop; never replace a draw.")
    return rows


def original_hashes(config):
    """Verify every original src module byte-for-byte against the frozen commit."""
    import hashlib
    if config["frozen_commit"] != FROZEN_COMMIT:
        raise ValueError("Unexpected frozen model commit.")
    for model, expected in (("B0", B0_HASH), ("B_revised", REVISED_HASH)):
        spec = config["models"][model]
        if spec["sha256"] != expected or sha256(ROOT/spec["config"]) != expected:
            raise ValueError(f"Frozen {model} configuration changed.")
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", FROZEN_COMMIT, "src"], cwd=ROOT, text=True).splitlines()
    result = {}
    for name in names:
        if not name.endswith(".py"):
            continue
        frozen = subprocess.check_output(["git", "show", f"{FROZEN_COMMIT}:{name}"], cwd=ROOT)
        digest = hashlib.sha256(frozen).hexdigest()
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Original model/reference code changed: {name}")
        result[name] = digest
    return result


def freeze(config, output):
    """Freeze protocol and original code BEFORE sampling new source parameters."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Existing benchmark is immutable; never silently overwrite.")
    code = original_hashes(config)
    legacy = yaml.safe_load((ROOT/config["models"]["B0"]["config"]).read_text())
    revised = yaml.safe_load((ROOT/config["models"]["B_revised"]["config"]).read_text())
    for key in ("domain", "physics", "measurement_times", "evaluation_times", "sensors_file", "sensor_count", "noise_std", "reference"):
        if config[key] != legacy[key]:
            raise ValueError(f"Controlled scientific setting differs: {key}")
    if config["optimization"]["primary_seed"] != revised["optimization_seed"] or config["recovery"] != {k:legacy["recovery"][k] for k in config["recovery"]}:
        raise ValueError("Frozen seed or recovery criterion changed.")
    if config["source_family"] != revised["source_family"]:
        raise ValueError("Source family changed.")
    for key in ("position_range", "strength_range"):
        if config["scenario_sampling"][key] != legacy["scenario_sampling"][key]:
            raise ValueError("Controlled source distribution changed.")
    code.update({config["models"][m]["config"]:config["models"][m]["sha256"] for m in config["models"]})
    code["src/inverpinn/data/fresh_blind.py"] = sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    (output/"protocol.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(ROOT/"docs/fresh_blind_benchmark_protocol.md", output/"protocol.md")
    shutil.copyfile(ROOT/config["sensors_file"], output/"sensor_geometry.json")
    for model in ("B0", "B_revised"):
        shutil.copyfile(ROOT/config["models"][model]["config"], output/f"{model}_frozen.yaml")
    run_provenance = provenance(output)
    pre = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=object_hash(config), protocol_document_hash=sha256(output/"protocol.md"),
        sensor_layout_hash=sha256(output/"sensor_geometry.json"),
        reference_solver_commit=FROZEN_COMMIT, reference_solver_hash=code["src/inverpinn/physics/finite_difference.py"],
        code_hashes=code, original_code_bundle_hash=object_hash(code),
        inference_config_hash=sha256(ROOT/config["reference"]["inference_config"]),
        model_config_hashes={k:config["models"][k]["sha256"] for k in config["models"]},
        initial_frozen_commit_verified=run_provenance["git_commit"] == FROZEN_COMMIT)
    write_json(output/"preregistration.json", pre)
    # Sampling follows the persisted timestamped protocol, not the reverse.
    plan = fresh_plan(config, legacy)
    write_json(output/"scenario_plan.json", plan)
    write_json(output/"protocol_lock.json", pre | dict(plan_hash=object_hash(plan)))
    write_json(output/"independence_check.json", dict(master_seed=config["scenario_sampling"]["seed"],
        fresh_scenarios=30, prior_source_hashes_checked=40, overlap=0,
        prior_data_or_outcomes_opened=False, method="Reconstruct hashes/seeds from frozen generation rules only"))


def manifest(root):
    """Hash-lock truth-bearing and sparse-only manifests before any fit."""
    root = Path(root)
    config, plan, lock = load_frozen(root)
    rows, inputs = [], []
    grid = next(c for c in config["reference"]["candidates"] if c["nx"] == config["reference"]["selected_grid"])
    for scenario in plan:
        case = fresh_path(root, scenario["scenario_id"])
        generation = json.loads((case/"generation.json").read_text())
        row = generation | {f"{k}_true":scenario["source"][v] for k,v in (("x","x_s"),("y","y_s"),("Q","Q"))}
        row.update(reference_grid=grid["nx"], dt=grid["dt"], steps=grid["steps"])
        rows.append(row)
        inputs.append(dict(scenario_id=row["scenario_id"], observations_hash=row["observations_hash"], physics_hash=row["physics_hash"]))
    lock_path = root/"manifest_lock.json"
    if lock_path.exists():
        stored = json.loads(lock_path.read_text())
        if object_hash(rows) != stored["manifest_hash"] or sha256(root/"scenario_manifest.csv") != stored["csv_hash"] or sha256(root/"input_manifest.json") != stored["inputs_hash"]:
            raise ValueError("Immutable fresh manifest changed.")
        return rows
    if (root/"scenario_manifest.csv").exists() or (root/"input_manifest.json").exists():
        raise FileExistsError("Partial manifests preserved; inspect rather than overwrite.")
    write_csv(root/"scenario_manifest.csv", rows)
    write_json(root/"input_manifest.json", inputs)
    write_json(lock_path, dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_hash=object_hash(rows), csv_hash=sha256(root/"scenario_manifest.csv"),
        inputs_hash=sha256(root/"input_manifest.json"), protocol_hash=lock["protocol_hash"]))
    return rows


def jobs(config):
    """60 primary jobs then 20 revised-only substudy jobs; never best-of-seeds."""
    seed = config["optimization"]["primary_seed"]
    primary = [(model, s, seed) for s in FRESH_IDS for model in ("B0", "B_revised")]
    extra = [("B_revised", s, k) for s in config["optimization"]["additional_seed_scenarios"]
             for k in config["optimization"]["additional_seeds"]]
    if len(extra) != 20 or seed in config["optimization"]["additional_seeds"] or len(set(primary+extra)) != 80:
        raise ValueError("Require exactly 60 primary and 20 distinct extra jobs.")
    return primary+extra


def verify(root):
    """Verify frozen protocol/code and copied model configs, with no old data."""
    root = Path(root)
    config, plan, lock = load_frozen(root)
    for model in ("B0", "B_revised"):
        if sha256(root/f"{model}_frozen.yaml") != config["models"][model]["sha256"]:
            raise ValueError(f"Frozen model copy changed: {model}")
    return config, plan, lock
