"""Revision-7-only LHS design, immutable manifests and consumed-data barriers."""

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

from inverpinn.data.diagnostic_v2 import diagnostic_plan
from inverpinn.data.fresh_blind import fresh_plan, original_hashes, REVISED_HASH
from inverpinn.data.single_source_benchmark import load_frozen, object_hash, scenario_plan, write_csv
from inverpinn.data.synthetic_reference import ROOT, provenance, sha256, write_json

IDS = tuple(f"development_v3_{i:03d}" for i in range(1, 25))
PREVIOUS_COMMIT = "199559f901bd21981f155d70550e3c82d06dde70"


def previous_plans():
    """Reconstruct only generation plans, never prior fit outcomes."""
    def read(name):
        return yaml.safe_load((ROOT/"configs"/name).read_text())
    legacy, fresh, diagnostic = [read(n) for n in
        ("single_source_benchmark.yaml", "fresh_blind_benchmark.yaml", "observability_diagnosis.yaml")]
    rows = scenario_plan(legacy)+fresh_plan(fresh, legacy)+diagnostic_plan(diagnostic, legacy, fresh)
    seeds = {legacy["scenario_sampling"][k]["seed"] for k in ("development", "heldout")}
    seeds |= {fresh["scenario_sampling"]["seed"], diagnostic["scenario_sampling"]["seed"]}
    return rows, seeds


def configured_sources_and_seeds():
    """Collect exclusion hashes from historical configs without displaying values."""
    sources, seeds = set(), set()
    def visit(value, key=""):
        if isinstance(value, dict):
            if {"x_s", "y_s", "Q"} <= value.keys():
                sources.add(object_hash({k:float(value[k]) for k in ("x_s", "y_s", "Q")}))
            for k, v in value.items():
                visit(v, k)
        elif isinstance(value, list):
            for v in value:
                visit(v, key)
        elif "seed" in key and type(value) is int:
            seeds.add(value)
    for path in (ROOT/"configs").glob("*.yaml"):
        if path.name not in ("inverse_formulation_revision.yaml", "B_revised_2.yaml"):
            visit(yaml.safe_load(path.read_text()))
    # Source truth/config files only, never metrics or estimates. This catches
    # early manually specified synthetic datasets outside revision plans.
    for path in (ROOT/"results").rglob("source.json"):
        if "truth" in path.parts and "inverse_formulation_revision" not in path.parts:
            visit(json.loads(path.read_text()))
    return sources, seeds


def development_plan(config):
    """One jittered sample per x/y/Q stratum; reject overlap, never redraw."""
    old, old_seeds = previous_plans()
    configured, configured_seeds = configured_sources_and_seeds()
    spec = config["scenario_sampling"]
    if spec["count"] != 24 or spec["seed"] in old_seeds | configured_seeds:
        raise ValueError("Need 24 cases and a previously unused master seed.")
    rng = np.random.default_rng(spec["seed"])
    unit = np.column_stack([(rng.permutation(24)+rng.random(24))/24 for _ in range(3)])
    lower = np.array([spec["position_range"][0]]*2+[spec["strength_range"][0]])
    upper = np.array([spec["position_range"][1]]*2+[spec["strength_range"][1]])
    children = np.random.SeedSequence(spec["seed"]).spawn(24)
    rows = []
    for identifier, values, child in zip(IDS, lower+(upper-lower)*unit, children):
        source = dict(zip(("x_s", "y_s", "Q"), map(float, values)))
        if object_hash(source) in configured:
            raise ValueError("Configured consumed source overlap; no replacement allowed.")
        source["sigma"] = config["source_family"]["sigma"]
        rows.append(dict(scenario_id=identifier, split="development_v3", source=source,
            source_hash=object_hash(source), generation_seed=int(child.generate_state(1, dtype=np.uint64)[0]) % (2**63-1)))
    for key in ("scenario_id", "source_hash", "generation_seed"):
        values = {r[key] for r in rows}
        if len(values) != 24 or values & {r[key] for r in old}:
            raise ValueError("Consumed plan overlap; no replacement allowed.")
    return rows


def inventory(root):
    """Opaque byte hashes for every historical result, including early pilots."""
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"results").rglob("*"))
            if p.is_file() and not p.is_relative_to(root)}


def verify_consumed(root):
    expected = json.loads((root/"consumed_hashes.json").read_text())
    if inventory(root) != expected:
        raise ValueError("Historical result files changed.")
    return len(expected)


def case_path(root, identifier, relative=""):
    if identifier not in IDS:
        raise ValueError("Only development_v3 IDs may enter this study.")
    lexical = Path(root).resolve()/"datasets"/identifier
    path = (lexical/relative).resolve()
    if lexical.resolve() != lexical or not path.is_relative_to(lexical):
        raise ValueError("Dataset path alias/escape forbidden.")
    return path


def check_open(root, path, fitting_case=None):
    """Accidental-I/O guard, not a security sandbox; fitting sees sparse inputs."""
    if isinstance(path, int):
        return
    root, path = Path(root).resolve(), Path(os.fsdecode(path)).resolve()
    if path.is_relative_to(ROOT/"results") and not path.is_relative_to(root):
        raise PermissionError("Consumed results forbidden in development workers.")
    if fitting_case is not None and path.is_relative_to(root):
        parts = path.relative_to(root).parts
        allowed = len(parts)==1 and parts[0] in {"protocol.yaml", "B_revised_frozen.yaml", "input_manifest.json", "sensor_geometry.json"}
        allowed |= path.is_relative_to(case_path(root, fitting_case, "inputs"))
        allowed |= len(parts)>=4 and parts[0]=="runs" and parts[1] in {"J0", "J1", "J2", "J3"} and parts[2]==fitting_case
        # J3 can read only a completed source estimate/checkpoint from J0, not
        # J0 metrics/truth. Per-candidate runner narrows these paths further.
        if len(parts)>=4 and parts[0]=="runs" and parts[-1] in {"metrics.json", "predictions.npz"}:
            allowed = False
        if not allowed:
            raise PermissionError("Truth/other scenarios cannot enter fitting.")


def install_guard(root):
    state = {"fitting_case":None}
    def audit(event, args):
        if event == "open":
            check_open(root, args[0], state["fitting_case"])
    sys.addaudithook(audit)
    return state


def freeze(config, root):
    if root.exists():
        raise FileExistsError("Revision 7 output already exists; no overwrite.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != PREVIOUS_COMMIT or config["preserved_revision_6_commit"] != head:
        raise ValueError("Revision 6 must be preserved first.")
    fresh = yaml.safe_load((ROOT/"configs/fresh_blind_benchmark.yaml").read_text())
    original_hashes(fresh)
    for key in ("domain", "physics", "source_family", "sensors_file", "sensor_count", "noise_std", "measurement_times", "evaluation_times", "reference", "recovery"):
        if config[key] != fresh[key]:
            raise ValueError(f"Controlled problem changed: {key}")
    code = {}
    for name in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", head, "src", "scripts"], cwd=ROOT, text=True).splitlines():
        if name.endswith(".py"):
            expected = hashlib.sha256(subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)).hexdigest()
            if sha256(ROOT/name) != expected:
                raise ValueError(f"Previous scientific code changed: {name}")
            code[name] = expected
    code["src/inverpinn/data/development_v3.py"] = sha256(Path(__file__))
    code["configs/B_revised.yaml"] = REVISED_HASH
    previous = inventory(root)
    root.mkdir(parents=True)
    (root/"protocol.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(ROOT/"docs/inverse_formulation_protocol.md", root/"protocol.md")
    shutil.copyfile(ROOT/"docs/source_field_coupling.md", root/"derivation.md")
    shutil.copyfile(ROOT/config["sensors_file"], root/"sensor_geometry.json")
    shutil.copyfile(ROOT/"configs/B_revised.yaml", root/"B_revised_frozen.yaml")
    provenance(root)
    pre = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), protocol_hash=object_hash(config),
        protocol_document_hash=sha256(root/"protocol.md"), derivation_hash=sha256(root/"derivation.md"),
        sensor_layout_hash=sha256(root/"sensor_geometry.json"), reference_solver_commit=head,
        reference_solver_hash=code["src/inverpinn/physics/finite_difference.py"], code_hashes=code,
        inference_config_hash=sha256(ROOT/config["reference"]["inference_config"]))
    write_json(root/"preregistration.json", pre)
    plan = development_plan(config)
    write_json(root/"scenario_plan.json", plan)
    write_json(root/"protocol_lock.json", pre | dict(plan_hash=object_hash(plan)))
    write_json(root/"consumed_hashes.json", previous)
    write_json(root/"independence_check.json", dict(master_seed=config["scenario_sampling"]["seed"],
        count=24, previous_planned_cases=100, overlap=0, historical_files_hashed=len(previous),
        prior_outcomes_used_for_selection=False))


def verify(root):
    config, plan, lock = load_frozen(root)
    if sha256(root/"B_revised_frozen.yaml") != REVISED_HASH or sha256(root/"derivation.md") != lock["derivation_hash"]:
        raise ValueError("Frozen model/derivation changed.")
    return config, plan, lock


def manifest(root):
    config, plan, lock = verify(root)
    rows, inputs = [], []
    for item in plan:
        case = case_path(root, item["scenario_id"])
        row = json.loads((case/"generation.json").read_text())
        for key, path in (("observations_hash", case/"inputs/observations.npz"), ("physics_hash", case/"inputs/physics.json"),
                          ("reference_data_hash", root/row["reference_path"])):
            if sha256(path) != row[key]:
                raise ValueError("Reference/input hash mismatch.")
        if row["max_reference_error_estimate"] is None or not np.isfinite(row["max_reference_error_estimate"]):
            raise ValueError("Undefined reference quality; preserve and review.")
        row.update(x_true=item["source"]["x_s"], y_true=item["source"]["y_s"], Q_true=item["source"]["Q"], reference_grid=641, dt=.00003125)
        rows.append(row)
        inputs.append({k:row[k] for k in ("scenario_id", "observations_hash", "physics_hash")})
    path = root/"manifest_lock.json"
    hashes = dict(manifest_hash=object_hash(rows), inputs_hash=object_hash(inputs))
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != v for k, v in hashes.items()) or sha256(root/"development_manifest.csv") != old["csv_hash"] or object_hash(json.loads((root/"input_manifest.json").read_text())) != old["inputs_hash"]:
            raise ValueError("Immutable manifest changed.")
        return rows
    if (root/"development_manifest.csv").exists() or (root/"input_manifest.json").exists():
        raise FileExistsError("Partial manifest preserved.")
    write_csv(root/"development_manifest.csv", rows)
    write_json(root/"input_manifest.json", inputs)
    write_json(path, hashes | dict(csv_hash=sha256(root/"development_manifest.csv"), frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    return rows
