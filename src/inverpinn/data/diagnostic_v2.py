"""New stratified diagnostic cases, immutable provenance and consumed-data guards.

Latin-hypercube marginals put one source coordinate/strength in every stratum.
This is a diagnostic design, not IID confirmatory sampling. Source truth is
excluded from input manifests and fitting APIs; explicit oracle diagnostics
are separate from inference. Original data/model implementations are untouched.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from inverpinn.data.fresh_blind import fresh_plan, original_hashes, REVISED_HASH
from inverpinn.data.single_source_benchmark import load_frozen, object_hash, scenario_plan, write_csv
from inverpinn.data.synthetic_reference import ROOT, provenance, sha256, write_json

IDS = tuple(f"diagnostic_v2_{i:03d}" for i in range(1, 31))
PRESERVED_COMMIT = "590c3faf0c6ec71677ca89b3093ea5a7c43116a7"
CONSUMED_ROOTS = ("single_source_benchmark", "model_diagnosis", "fresh_blind_benchmark")


def diagnostic_plan(config, legacy, fresh):
    """One independently jittered/permuted stratum per marginal, no selection."""
    spec = config["scenario_sampling"]
    old = scenario_plan(legacy)+fresh_plan(fresh, legacy)
    forbidden = {fresh["scenario_sampling"]["seed"],
                 *(legacy["scenario_sampling"][k]["seed"] for k in ("development", "heldout"))}
    if spec["count"] != 30 or spec["seed"] in forbidden:
        raise ValueError("Require 30 diagnostic scenarios and a new master seed.")
    rng = np.random.default_rng(spec["seed"])
    unit = np.column_stack([(rng.permutation(30)+rng.random(30))/30 for _ in range(3)])
    lower = np.array([spec["position_range"][0]]*2+[spec["strength_range"][0]])
    upper = np.array([spec["position_range"][1]]*2+[spec["strength_range"][1]])
    sources = lower+(upper-lower)*unit
    children = np.random.SeedSequence(spec["seed"]).spawn(30)
    rows = []
    for identifier, values, child in zip(IDS, sources, children):
        source = dict(zip(("x_s", "y_s", "Q"), map(float, values))) | {"sigma":config["source_family"]["sigma"]}
        seed = int(child.generate_state(1, dtype=np.uint64)[0]) % (2**63-1)
        rows.append(dict(scenario_id=identifier, split="diagnostic_v2", generation_seed=seed,
                         source=source, source_hash=object_hash(source)))
    for key in ("scenario_id", "generation_seed", "source_hash"):
        current = {r[key] for r in rows}
        if len(current) != 30 or current & {r[key] for r in old}:
            raise ValueError("Diagnostic/consumed overlap; never replace a draw.")
    return rows


def diagnostic_path(root, identifier, relative=""):
    """Only new allowlisted IDs; reject traversal and symlink aliasing."""
    if identifier not in IDS:
        raise ValueError("Only diagnostic_v2_001 through diagnostic_v2_030 allowed.")
    lexical = Path(root).resolve()/"datasets"/identifier
    case, path = lexical.resolve(), (lexical/relative).resolve()
    if case != lexical or not path.is_relative_to(case):
        raise ValueError("Diagnostic path escapes or aliases a dataset.")
    return path


def check_open(root, path, fitting_case=None):
    """Block consumed results and truth access inside classical/PINN fitting.

    This protects against accidental Python I/O, not malicious OS-level access.
    Reference subprocesses receive only newly written diagnostic configs.
    """
    if isinstance(path, int):
        return
    root, path = Path(root).resolve(), Path(os.fsdecode(path)).resolve()
    if path.is_relative_to((ROOT/"results").resolve()) and not path.is_relative_to(root):
        raise PermissionError("Consumed results are forbidden in diagnostic workers.")
    if fitting_case is not None and path.is_relative_to(root):
        parts = path.relative_to(root).parts
        allowed_files = {"protocol.yaml", "B_revised_frozen.yaml", "sensor_geometry.json", "input_manifest.json", "manifest_lock.json"}
        allowed = (len(parts)==1 and parts[0] in allowed_files)
        allowed |= path.is_relative_to(diagnostic_path(root, fitting_case, "inputs"))
        allowed |= len(parts)>=3 and parts[0]=="runs" and parts[1] in ("pinn", "classical") and parts[2]==fitting_case
        if not allowed:
            raise PermissionError("Fitting may not open truth, oracle outputs or other scenarios.")


def install_guard(root):
    state = {"fitting_case":None}
    def audit(event, args):
        if event == "open":
            check_open(root, args[0], state["fitting_case"])
    sys.addaudithook(audit)
    return state


def consumed_inventory():
    """Hash opaque historical bytes for preservation, never parse outcomes."""
    return {str(p.relative_to(ROOT)):sha256(p)
            for name in CONSUMED_ROOTS for p in sorted((ROOT/"results"/name).rglob("*")) if p.is_file()}


def verify_consumed(root):
    """Call outside worker I/O guards; check historical files without analysis."""
    expected = json.loads((Path(root)/"consumed_hashes.json").read_text())
    if consumed_inventory() != expected:
        raise ValueError("Consumed benchmark inventory changed.")
    return len(expected)


def freeze(config, root):
    """Persist scientific choices before LHS sampling or reference generation."""
    root = Path(root)
    if root.exists():
        raise FileExistsError("Existing diagnostic directory is immutable.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != PRESERVED_COMMIT or config["preserved_revision_5_commit"] != PRESERVED_COMMIT:
        raise ValueError("Revision 5 must first be preserved at its expected commit.")
    old_config = yaml.safe_load((ROOT/"configs/fresh_blind_benchmark.yaml").read_text())
    legacy = yaml.safe_load((ROOT/"configs/single_source_benchmark.yaml").read_text())
    original_hashes(old_config)
    for key in ("domain", "physics", "source_family", "sensors_file", "sensor_count", "noise_std", "measurement_times", "evaluation_times", "reference", "recovery"):
        if config[key] != old_config[key]:
            raise ValueError(f"Controlled setting changed: {key}")
    if config["models"]["B_revised"] != old_config["models"]["B_revised"] or config["optimization"]["primary_seed"] != 42:
        raise ValueError("Frozen revised model or optimization seed changed.")
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", head, "src", "scripts"], cwd=ROOT, text=True).splitlines()
    code = {}
    for name in names:
        if name.endswith(".py"):
            expected = hashlib.sha256(subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)).hexdigest()
            if sha256(ROOT/name) != expected:
                raise ValueError(f"Original source changed: {name}")
            code[name] = expected
    code["src/inverpinn/data/diagnostic_v2.py"] = sha256(Path(__file__))
    code["configs/B_revised.yaml"] = REVISED_HASH
    old_inventory = consumed_inventory()
    root.mkdir(parents=True)
    (root/"protocol.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(ROOT/"docs/observability_diagnosis_protocol.md", root/"protocol.md")
    shutil.copyfile(ROOT/config["sensors_file"], root/"sensor_geometry.json")
    shutil.copyfile(ROOT/config["models"]["B_revised"]["config"], root/"B_revised_frozen.yaml")
    provenance(root)
    pre = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), protocol_hash=object_hash(config),
        protocol_document_hash=sha256(root/"protocol.md"), sensor_layout_hash=sha256(root/"sensor_geometry.json"),
        reference_solver_commit=head, reference_solver_hash=code["src/inverpinn/physics/finite_difference.py"],
        code_hashes=code, inference_config_hash=sha256(ROOT/config["reference"]["inference_config"]))
    write_json(root/"preregistration.json", pre)
    plan = diagnostic_plan(config, legacy, old_config)
    write_json(root/"scenario_plan.json", plan)
    write_json(root/"protocol_lock.json", pre | dict(plan_hash=object_hash(plan)))
    write_json(root/"consumed_hashes.json", old_inventory)
    write_json(root/"independence_check.json", dict(master_seed=config["scenario_sampling"]["seed"],
        diagnostic_count=30, consumed_source_count=70, overlap=0, design="permuted jittered Latin hypercube",
        consumed_outcomes_parsed=False, historical_files_hashed=len(old_inventory)))


def verify(root):
    config, plan, lock = load_frozen(root)
    if sha256(Path(root)/"B_revised_frozen.yaml") != REVISED_HASH:
        raise ValueError("Frozen B_revised copy changed.")
    return config, plan, lock


def manifest(root):
    """Freeze all generated references and sparse-only inputs before diagnosis."""
    root = Path(root)
    config, plan, lock = verify(root)
    rows, inputs = [], []
    for item in plan:
        case = diagnostic_path(root, item["scenario_id"])
        row = json.loads((case/"generation.json").read_text())
        if row["max_reference_error_estimate"] is None or not np.isfinite(row["max_reference_error_estimate"]):
            raise RuntimeError("Undefined reference accuracy; preserved case requires review.")
        for name, path in (("observations_hash", case/"inputs/observations.npz"),
                           ("physics_hash", case/"inputs/physics.json"),
                           ("reference_data_hash", root/row["reference_path"])):
            if sha256(path) != row[name]:
                raise ValueError("Generated reference/input hash changed.")
        row.update(x_true=item["source"]["x_s"], y_true=item["source"]["y_s"], Q_true=item["source"]["Q"],
                   reference_grid=641, dt=.00003125, steps=25600)
        rows.append(row)
        inputs.append({k:row[k] for k in ("scenario_id", "observations_hash", "physics_hash")})
    path = root/"manifest_lock.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        if (object_hash(rows) != frozen["manifest_hash"] or sha256(root/"diagnostic_manifest.csv") != frozen["csv_hash"]
                or sha256(root/"input_manifest.json") != frozen["inputs_hash"]):
            raise ValueError("Immutable diagnostic manifest changed.")
        return rows
    if (root/"diagnostic_manifest.csv").exists() or (root/"input_manifest.json").exists():
        raise FileExistsError("Partial diagnostic manifest preserved.")
    write_csv(root/"diagnostic_manifest.csv", rows)
    write_json(root/"input_manifest.json", inputs)
    write_json(path, dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), manifest_hash=object_hash(rows),
        csv_hash=sha256(root/"diagnostic_manifest.csv"), inputs_hash=sha256(root/"input_manifest.json"),
        protocol_hash=lock["protocol_hash"]))
    return rows


def require_information_complete(root):
    """PINN cannot start before all diagnostic artifacts are persisted/frozen."""
    root = Path(root)
    path = root/"information_complete.json"
    if not path.exists():
        raise RuntimeError("Information diagnostics must complete before PINN fitting.")
    record = json.loads(path.read_text())
    for name, digest in record["hashes"].items():
        if sha256(root/name) != digest:
            raise ValueError("Frozen information-stage artifact changed.")
    return record
