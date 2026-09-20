"""Prespecified oracle-fixed-Q and observation-profiled-Q maps for fresh_015."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import torch
import yaml

from inverpinn.data.fresh_blind import fresh_path, install_guard, verify
from inverpinn.data.synthetic_reference import ROOT, inference_observations, load_model_inputs, write_json
from inverpinn.evaluation.fresh_blind import profile_strength
from inverpinn.data.single_source_benchmark import write_csv


def location_response(root, x, y):
    """Forward map uses candidate coordinates and Q=1, never the true Q."""
    config = yaml.safe_load((root/"protocol.yaml").read_text())
    inputs = load_model_inputs(fresh_path(root, config["identifiability"]["scenario_id"]))
    inference = yaml.safe_load((ROOT/config["reference"]["inference_config"]).read_text())
    reference = config["reference"]["candidates"][-1]
    torch.set_num_threads(1)
    values = inference_observations(dict(x_s=float(x), y_s=float(y), Q=1., sigma=config["source_family"]["sigma"]),
                                    inputs, inference, reference).numpy()
    return values


def run(root):
    install_guard(root)
    config, _, _ = verify(root)
    if not (root/"fits_complete.json").exists():
        raise RuntimeError("Identifiability is post-fit only.")
    directory = root/"identifiability"
    directory.mkdir(exist_ok=False)
    spec = config["identifiability"]
    (directory/"config.yaml").write_text(yaml.safe_dump(spec | dict(inference=yaml.safe_load((ROOT/config["reference"]["inference_config"]).read_text()))))
    case = fresh_path(root, spec["scenario_id"])
    inputs = load_model_inputs(case)
    # Oracle truth is used only to form diagnostic A and labeled extra probes.
    truth = json.loads((case/"truth/source.json").read_text())
    run_dir = root/"runs/B_revised"/spec["scenario_id"]/f"seed_{config['optimization']['primary_seed']}"
    predicted = json.loads((run_dir/"source_estimate.json").read_text()) if (run_dir/"source_estimate.json").exists() else None
    axis = np.linspace(*spec["position_range"], spec["grid_count"])
    points = [(float(x), float(y)) for y in axis for x in axis]
    probes = [("true_location_oracle_probe", truth["x_s"], truth["y_s"])]
    if predicted:
        probes.append(("primary_PINN_location_probe", predicted["x_s"], predicted["y_s"]))
    observations = inputs["measurements"].numpy()
    responses, rows = [], []
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"], initializer=install_guard, initargs=(root,)) as pool:
        futures = [pool.submit(location_response, root, x, y) for x,y in points+[(x,y) for _,x,y in probes]]
        for i, future in enumerate(futures):
            g = future.result()
            q, mse = profile_strength(g, observations)
            x, y = (points+[(x,y) for _,x,y in probes])[i]
            rows.append(dict(kind="grid" if i < len(points) else probes[i-len(points)][0],
                x=x, y=y, fixed_true_Q=truth["Q"], fixed_Q_mse=float(np.mean((truth["Q"]*g-observations)**2)),
                profiled_Q=q, profiled_Q_mse=mse))
            responses.append(g)
            if (i+1) % len(axis) == 0:
                print(f"IDENTIFIABILITY rows complete: {(i+1)//len(axis)}/{len(axis)}", flush=True)
    write_csv(directory/"landscapes.csv", rows)
    np.savez_compressed(directory/"unit_responses.npz", unit_readings=np.stack(responses), observed=observations,
                        coordinates=np.array(points+[(x,y) for _,x,y in probes]), axis=axis)
    xx, yy = np.meshgrid(axis, axis)
    distance = np.hypot(xx-truth["x_s"], yy-truth["y_s"])
    summary = dict(scenario_id=spec["scenario_id"], true_source=truth, primary_PINN_source=predicted,
                   oracle_fixed_Q=True, profiled_Q_uses_truth=False, posterior_probability=False,
                   reference_grid=641, diagnostic_solver_grid=201, extra_probes=rows[len(points):])
    for name in ("fixed_Q", "profiled_Q"):
        values = np.array([r[f"{name}_mse"] for r in rows[:len(points)]]).reshape(xx.shape)
        j,i = np.unravel_index(np.argmin(values), values.shape)
        summary[name] = dict(minimum_mse=float(values[j,i]), minimum_location=[float(axis[i]),float(axis[j])],
            minimum_localization_error=float(distance[j,i]),
            Q_at_grid_minimum=rows[j*len(axis)+i]["profiled_Q"] if name == "profiled_Q" else truth["Q"],
            minimum_mse_outside_one_sigma=float(values[distance > config["source_family"]["sigma"]].min()),
            minimum_mse_outside_two_sigma=float(values[distance > 2*config["source_family"]["sigma"]].min()))
    write_json(directory/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fresh_blind_benchmark.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    run(ROOT/config["output_dir"])
