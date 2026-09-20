"""Preregister and generate final-blind references, never tune a model."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from inverpinn.data.final_confirmatory import freeze,verify,manifest,install_guard,read_config
from inverpinn.data.single_source_benchmark import generate_scenario
from inverpinn.data.synthetic_reference import ROOT


def generate(root):
    config,plan,_=verify(root);install_guard(root)
    todo=[]
    for row in plan:
        case=root/"datasets"/row["scenario_id"]
        if (case/"generation.json").exists():continue
        if case.exists():raise RuntimeError(f"Partial reference preserved: {case}")
        todo.append(row)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"],initializer=install_guard,initargs=(root,)) as pool:
        for f in as_completed([pool.submit(generate_scenario,root,row) for row in todo]):
            row=f.result();print(f"REFERENCE {row['scenario_id']}: target={row['reference_target_met']}",flush=True)
    rows=manifest(root)
    print(f"MANIFEST LOCKED {len(rows)}; target misses={sum(not r['reference_target_met'] for r in rows)}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("freeze","generate"),required=True)
    args=parser.parse_args();config=read_config("final_confirmatory.yaml");root=ROOT/config["output_dir"]
    freeze(config,root) if args.stage=="freeze" else generate(root)
