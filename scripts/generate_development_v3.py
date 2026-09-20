"""Freeze and generate new Revision-7 development data without consuming outcomes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

from inverpinn.data.development_v3 import freeze, install_guard, manifest, verify
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
            raise RuntimeError(f"Incomplete reference preserved: {case}")
        todo.append(row)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"], initializer=install_guard, initargs=(root,)) as pool:
        for future in as_completed([pool.submit(generate_scenario, root, row) for row in todo]):
            row = future.result()
            print(f"REFERENCE COMPLETE {row['scenario_id']} target_met={row['reference_target_met']}", flush=True)
    rows = manifest(root)
    print(f"MANIFEST FROZEN: {len(rows)} cases; {sum(not r['reference_target_met'] for r in rows)} target misses", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "generate"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/"configs/inverse_formulation_revision.yaml").read_text())
    root = ROOT/config["output_dir"]
    freeze(config, root) if args.stage == "freeze" else generate(root)
