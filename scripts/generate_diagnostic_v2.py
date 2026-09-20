"""Freeze or generate only the new revision-6 stratified diagnostic set."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import yaml

from inverpinn.data.diagnostic_v2 import freeze, install_guard, manifest, verify
from inverpinn.data.single_source_benchmark import generate_scenario
from inverpinn.data.synthetic_reference import ROOT


def generate(root):
    install_guard(root)
    config, plan, _ = verify(root)
    todo = []
    for row in plan:
        case = root/"datasets"/row["scenario_id"]
        if (case/"generation.json").exists():
            continue
        if case.exists():
            raise RuntimeError(f"Incomplete reference preserved, do not overwrite: {case}")
        todo.append(row)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"], initializer=install_guard, initargs=(root,)) as pool:
        futures = {pool.submit(generate_scenario, root, item):item["scenario_id"] for item in todo}
        for future in as_completed(futures):
            row = future.result()
            print(f"REFERENCE COMPLETE {row['scenario_id']}; target_met={row['reference_target_met']}; estimate={row['max_reference_error_estimate']}", flush=True)
    rows = manifest(root)
    print(json.dumps(dict(generated=len(rows), target_misses=sum(not r["reference_target_met"] for r in rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/observability_diagnosis.yaml")
    parser.add_argument("--stage", choices=("freeze", "generate"), required=True)
    args = parser.parse_args()
    settings = yaml.safe_load((ROOT/args.config).read_text())
    root = ROOT/settings["output_dir"]
    freeze(settings, root) if args.stage == "freeze" else generate(root)
