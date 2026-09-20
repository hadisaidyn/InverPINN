"""Freeze revision-5 protocol, then generate all 30 untouched references."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from inverpinn.data.fresh_blind import freeze, install_guard, manifest
from inverpinn.data.single_source_benchmark import generate_scenario, load_frozen
from inverpinn.data.synthetic_reference import ROOT


def generate(root):
    config, plan, _ = load_frozen(root)
    pending = []
    for scenario in plan:
        case = root/"datasets"/scenario["scenario_id"]
        if (case/"generation.json").exists():
            continue
        if case.exists():
            raise FileExistsError(f"Incomplete reference preserved; no replacement/retry: {case}")
        pending.append(scenario)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"], initializer=install_guard, initargs=(root,)) as pool:
        tasks = {pool.submit(generate_scenario, root, s):s["scenario_id"] for s in pending}
        for task in as_completed(tasks):
            row = task.result()
            print(f"REFERENCE {row['scenario_id']} complete; target_met={row['reference_target_met']}", flush=True)
    rows = manifest(root)
    if any(r["max_reference_error_estimate"] is None for r in rows):
        raise RuntimeError("Undefined Richardson estimate: preserved benchmark requires scientific review before fitting.")
    print(f"MANIFEST FROZEN: {len(rows)} cases, {sum(not r['reference_target_met'] for r in rows)} flagged target misses", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fresh_blind_benchmark.yaml")
    parser.add_argument("--stage", choices=("freeze", "generate"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    root = ROOT/config["output_dir"]
    install_guard(root)
    if args.stage == "freeze":
        freeze(config, root)
        print("Protocol frozen; no reference simulation or model evaluation performed.")
    else:
        generate(root)
