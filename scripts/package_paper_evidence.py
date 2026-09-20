"""Copy a compact, traceable reporting bundle from the completed local archive.

This is packaging, not an experiment: no model, solver, optimizer or checkpoint
is loaded. Every copy is byte-identical. All original data/results and frozen
scientific source/config files are SHA-256 inventoried before packaging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "eb18380f1488f5c679a0981da75e667fa3ba1b51"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_inventory():
    paths = set()
    for folder in ("results", "data", "src", "configs"):
        paths.update(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    # Original scripts/tests must not be changed by reporting-only finalization.
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", FROZEN, "scripts", "tests"], cwd=ROOT, text=True)
    paths.update(ROOT / p for p in tracked.splitlines())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/evidence")
    parser.add_argument("--verify-preservation", action="store_true")
    args = parser.parse_args()
    inventory_path = ROOT / "artifacts/finalization_audit/protected_before.json"
    if args.verify_preservation:
        before = json.loads(inventory_path.read_text())
        after = protected_inventory()
        changed = [p for p in before if before[p] != after.get(p)]
        added = sorted(set(after) - set(before))
        if changed or added:
            raise RuntimeError(f"Protected archive changed: changed={changed}, added={added}")
        print(f"PASS: all {len(before)} protected files unchanged")
        return
    if args.output_dir.exists() or inventory_path.exists():
        raise FileExistsError("Packaging/audit output already exists; no overwrite permitted")
    inventory = protected_inventory()
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    final = "results/final_confirmatory/"
    sources = {
        "final/" + n: final + n for n in (
            "J1_raw.csv", "B_revised_raw.csv", "classical_raw.csv", "manifest.csv",
            "observability.csv", "information_strata.csv", "decision.json", "summary.json",
            "protocol.yaml", "protocol.md", "protocol_lock.json", "manifest_lock.json",
            "preregistration.json", "independence_check.json", "integrity_audit.json",
            "environment.json", "sensor_geometry.json", "provenance.json",
            "paired_J1_vs_Brevised.csv", "paired_J1_vs_classical.csv",
            "final_targeted_tests.xml", "final_full_suite.xml")}
    for name in ("numerical_convergence.csv", "validation_summary.json", "config.yaml"):
        sources["numerical/" + name] = "results/numerical_validation/" + name
    sources["historical/initial_summary.json"] = "results/single_source_benchmark/summary.json"
    sources["historical/physical_correction_summary.json"] = "results/fresh_blind_benchmark/summary.json"
    for name in ("summary.json", "conditional_Q_results.csv", "resolution_bias_results.csv", "map_metrics.csv"):
        sources["observability/" + name] = "results/observability_diagnosis/" + name
    for name in ("candidate_summary.csv", "Q_bias_summary.csv", "summary.json"):
        sources["development/" + name] = "results/inverse_formulation_revision/" + name
    for name in ("J1_confirmatory.yaml", "B_revised.yaml", "final_confirmatory.yaml",
                 "reference_inference.yaml", "single_source_sensors.json"):
        sources["configs/" + name] = "configs/" + name
    j1 = pd.read_csv(ROOT / final / "J1_raw.csv")
    recovered = j1[j1.recovery_success].sort_values(["localization_error", "scenario_id"])
    selected = {
        "success": recovered.iloc[len(recovered) // 2].scenario_id,
        "failure": j1.sort_values(["localization_error", "scenario_id"],
                                 ascending=[False, True]).iloc[0].scenario_id}
    for role, scenario in selected.items():
        sources[f"previews/{role}.npz"] = final + f"runs/J1/{scenario}/predictions.npz"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for dest, source in sources.items():
        src, target = ROOT / source, args.output_dir / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        entries[dest] = {"source": source, "sha256": sha256(target),
                         "bytes": target.stat().st_size, "transformation": "byte-identical copy"}
        if entries[dest]["sha256"] != sha256(src):
            raise RuntimeError(f"Copy verification failed: {source}")
    manifest = dict(frozen_scientific_commit=FROZEN, files=entries, selected_cases=selected,
        selection_rule={"success": "median rank (floor(n/2), zero-based) among joint recoveries, localization ascending then ID",
                        "failure": "largest localization error, ID ascending for ties"},
        preview_note="Four saved 161x161 stride-4 previews; quantitative metrics use full 641x641 fields.",
        protected_file_count=len(inventory), protected_inventory_sha256=sha256(inventory_path))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Packaged {len(entries)} unchanged files; selected {selected}; protected {len(inventory)} files")


if __name__ == "__main__":
    main()
