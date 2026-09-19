"""Selective reference infrastructure, isolated from training and inverse fitting.

Schema 2 deliberately separates inputs/ (no source truth) from truth/ and
provenance. Existing schema-1 scripts are not silently adapted or retrained.
"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

from inverpinn.data.sensors import sample_sensors
from inverpinn.evaluation.reference import choose_reference, compare_references
from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number

ROOT = Path(__file__).resolve().parents[3]
SCHEME = "forward Euler; first-order upwind advection; centered diffusion; CPU float64"
UNITS = dict(x="synthetic length", y="synthetic length", t="synthetic time", C="synthetic concentration",
             u="length/time", v="length/time", D="length^2/time", Q="concentration/time", sigma="length")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(output):
    """Freeze actual working code, not just a potentially dirty Git commit."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Git provenance unavailable: {result.stderr}")
        return result.stdout.strip()
    names = ["src/inverpinn/physics/finite_difference.py", "src/inverpinn/physics/sources.py",
        "src/inverpinn/physics/manufactured.py", "src/inverpinn/data/sensors.py",
        "src/inverpinn/data/synthetic_reference.py", "src/inverpinn/evaluation/reference.py",
        "scripts/benchmark_reference.py", "scripts/generate_reference_pilot.py"]
    with zipfile.ZipFile(output/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name, name)
    result = dict(git_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        working_tree_status=git("status", "--short"), generated_at_utc=datetime.now(timezone.utc).isoformat(),
        solver_type=SCHEME, source_hashes={name: sha256(ROOT/name) for name in names},
        source_archive_sha256=sha256(output/"source_snapshot.zip"))
    write_json(output/"provenance.json", result)
    write_json(output/"environment.json", dict(python=platform.python_version(), platform=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}))
    return result


def observation_coordinates(config):
    """Return deterministic fixed positions and physical measurement times.

    A local seed controls placement without changing global RNG state. Random
    positions are continuous, never snapped to any reference/inference nodes.
    """
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**63-1:
        raise ValueError("seed must be an integer in [0,2**63-2].")
    settings = config["measurement_times"]
    if type(settings["count"]) is not int or settings["count"] < 2:
        raise ValueError("At least two measurement times are required.")
    times = torch.linspace(settings["start"], settings["stop"], settings["count"], dtype=torch.float64)
    sensors = config["sensors"]
    kwargs = dict(positions=sensors["positions"]) if "positions" in sensors else dict(n_sensors=sensors["count"])
    axis = torch.tensor([0, 1], dtype=torch.float64)
    positions, _ = sample_sensors(torch.zeros((1, 2, 2), dtype=torch.float64), axis, axis,
                                  seed=config["seed"], **kwargs)
    return positions, times


def validate_separation(reference, inference, physics, domain):
    """Reject identical grids and shared-source inference configurations.

    Distinct grids define distinct discrete transport matrices and bilinear
    observation matrices, even though the continuous PDE/interpolation family
    are shared. This mitigates the same-grid inverse crime, not shared-physics
    model error. No source truth is an inference solver setting.
    """
    required = {"nx", "ny", "dt", "steps"}
    if set(inference["solver"]) != required:
        raise ValueError("Inference solver must contain only nx,ny,dt,steps; no sources.")
    if domain != dict(x=[0.0, 1.0], y=[0.0, 1.0]) or inference["domain"] != domain:
        raise ValueError("Reference and inference require the same unit-square domain.")
    other = inference["solver"]
    if (reference["nx"], reference["ny"]) == (other["nx"], other["ny"]):
        raise ValueError("Reference and inference grids must differ; same-grid inverse crime is forbidden.")
    for key in ("nx", "ny"):
        if type(other[key]) is not int or other[key] < 3:
            raise ValueError("Inference grid sizes must be integers >=3.")
    if type(other["steps"]) is not int or other["steps"] < 1:
        raise ValueError("Inference steps must be a positive integer.")
    if not math.isclose(reference["dt"]*reference["steps"], other["dt"]*other["steps"], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Reference and inference must span the same physical interval.")
    expected = dict(spatial="bilinear", temporal="linear", coordinates="physical", snap_to_nodes=False)
    if inference["observation"] != expected:
        raise ValueError("Use separately evaluated bilinear/linear physical-coordinate observation maps, without snapping.")
    cfl = stability_number(dx=1/(other["nx"]-1), dy=1/(other["ny"]-1), dt=other["dt"], **physics)
    if cfl > 1:
        raise ValueError("Inference timestep violates the combined CFL bound.")
    return dict(reference_solver=reference, inference_solver=other, inference_CFL=cfl,
        observation_mapping="Bilinear weights recomputed on each grid; linear time brackets recomputed for each dt; identical physical sensor coordinates/times.",
        limitation="Same PDE and stencil family; this prevents identical-grid synthetic fitting, not all forms of inverse crime.")


def inference_observations(candidate_source, inputs, inference, reference_solver):
    """Future trial-source forward map only; no optimizer or source-truth loading."""
    validate_separation(reference_solver, inference, inputs["physics"]["transport"], inputs["physics"]["domain"])
    result = solve_advection_diffusion(**inference["solver"], **inputs["physics"]["transport"],
        sources=[candidate_source], output_mode="sensor_only", sensor_positions=inputs["positions"],
        measurement_times=inputs["times"].tolist())
    return result["measurements"]


def load_model_inputs(scenario):
    """Open ONLY two allowlisted input files; never metadata, config, or truth.

    Features have columns [x,y,t]. Targets are noisy concentration. Source
    coordinates, Q, sigma, clean readings and dense fields are not returned.
    This prevents accidental loader leakage, not deliberate access to truth/.
    """
    directory = Path(scenario)/"inputs"
    with np.load(directory/"observations.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"positions", "times", "measurements"}:
            raise ValueError("Model-input archive has unexpected fields; source/evaluation truth is forbidden.")
        positions, times, readings = [torch.from_numpy(stored[k].copy()) for k in ("positions", "times", "measurements")]
    physics = json.loads((directory/"physics.json").read_text())
    if set(physics) != {"transport", "domain", "initial_condition", "boundary_condition", "units"} or set(physics["transport"]) != {"u", "v", "D"}:
        raise ValueError("Model-input physics must not contain source truth or additional features.")
    if readings.shape != (len(times), len(positions)) or positions.shape != (len(positions), 2):
        raise ValueError("Invalid sparse observation shapes.")
    if any(not torch.isfinite(a).all() for a in (positions, times, readings)):
        raise ValueError("Nonfinite model inputs.")
    features = torch.cat((positions.repeat(len(times), 1), times.repeat_interleave(len(positions))[:, None]), 1)
    return dict(positions=positions, times=times, measurements=readings, features=features,
                targets=readings.reshape(-1, 1), physics=physics)


def _rss_bytes():
    """Process high-water RSS, including Python/PyTorch, not tensor bytes alone."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def run_case(config, output):
    """Isolated worker: run one grid and save only requested clean outputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.yaml").write_text(yaml.safe_dump(config))
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    positions = torch.tensor(config["positions"], dtype=torch.float64)
    baseline = _rss_bytes()
    start = time.perf_counter()
    result = solve_advection_diffusion(**config["solver"], **config["physics"], sources=[config["source"]],
        output_mode="selected_times", output_times=config["evaluation_times"],
        sensor_positions=positions, measurement_times=config["times"])
    runtime = time.perf_counter()-start
    peak = _rss_bytes()
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite selective reference output.")
    np.savez_compressed(output/"checkpoint.npz", **{k: v.numpy() for k,v in result.items()})
    solver = config["solver"]
    metrics = dict(grid_size=solver["nx"], **solver, dx=1/(solver["nx"]-1), dy=1/(solver["ny"]-1),
        runtime_seconds=runtime, peak_process_rss_bytes=peak, pre_solve_high_water_rss_bytes=baseline,
        rss_high_water_increase_bytes=max(0, peak-baseline),
        retained_field_bytes=result["fields"].numel()*8,
        retained_measurement_bytes=result["measurements"].numel()*8,
        hypothetical_full_history_bytes=(solver["steps"]+1)*solver["nx"]*solver["ny"]*8,
        CFL=stability_number(dx=1/(solver["nx"]-1), dy=1/(solver["ny"]-1), dt=solver["dt"], **config["physics"]),
        checkpoint_sha256=sha256(output/"checkpoint.npz"),
        memory_method="Fresh subprocess high-water RSS includes runtime/imports; increase is not an exact tensor-only peak.")
    write_json(output/"metrics.json", metrics)
    return metrics


def run_refinement(config, output):
    """Run three independently timed/memory-profiled grids, without fitting."""
    import csv
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    positions, times = observation_coordinates(config)
    inference_path = ROOT/config["inference_config"]
    inference = yaml.safe_load(inference_path.read_text())
    candidates = config["candidates"]
    if len(candidates) != 3 or any(b["nx"]-1 != 2*(a["nx"]-1) for a,b in zip(candidates,candidates[1:])):
        raise ValueError("Use exactly three successively doubled candidate grids.")
    final_times = [c["dt"]*c["steps"] for c in candidates]
    if any(not math.isclose(t, final_times[0], rel_tol=1e-12) for t in final_times):
        raise ValueError("All candidate simulations must have the same final time.")
    if config["evaluation_times"][-1] != final_times[0]:
        raise ValueError("The final field must be included exactly once among evaluation_times.")
    for candidate in candidates:
        if candidate["nx"] != candidate["ny"]:
            raise ValueError("Reference refinement requires square grids.")
        validate_separation(candidate, inference, config["physics"], config["domain"])
    resolved = config | dict(positions=positions.tolist(), times=times.tolist(), inference_configuration=inference)
    (output/"config.yaml").write_text(yaml.safe_dump(resolved))
    provenance(output)
    measurements, cases = [], []
    for candidate in candidates:
        case_dir = output/f"grid_{candidate['nx']}"
        case_config = {k:resolved[k] for k in ("seed", "torch_threads", "source", "physics", "evaluation_times", "positions", "times")}
        case_config["solver"] = candidate
        # Input outside the worker's new directory, so the worker can enforce
        # non-overwrite protection itself. Each run retains this exact config.
        path = output/f"grid_{candidate['nx']}.yaml"
        path.write_text(yaml.safe_dump(case_config))
        process = subprocess.run([sys.executable, "-m", "inverpinn.data.synthetic_reference",
            str(path), str(case_dir)], capture_output=True, text=True)
        if process.returncode:
            (output/"failure.txt").write_text(process.stdout+process.stderr)
            raise RuntimeError(f"Reference grid {candidate['nx']} failed: {process.stderr}")
        metrics = json.loads((case_dir/"metrics.json").read_text())
        measurements.append(metrics)
        with np.load(case_dir/"checkpoint.npz", allow_pickle=False) as arrays:
            cases.append({k: torch.from_numpy(arrays[k].copy()) for k in arrays.files})
        print(f"Reference {candidate['nx']}x{candidate['ny']}: {metrics['runtime_seconds']:.2f}s, peak RSS {metrics['peak_process_rss_bytes']/1e6:.1f} MB", flush=True)
    comparisons = compare_references(cases)
    by_grid = {m["grid_size"]:m for m in measurements}
    rows = [by_grid[r["grid_size"]] | r for r in comparisons]
    with (output/"reference_benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    summary = choose_reference(comparisons, {m["grid_size"]:m["runtime_seconds"] for m in measurements}, config["target_relative_L2"])
    summary.update(candidates=measurements, evaluation_times=config["evaluation_times"],
        measurement_count=len(times), sensor_count=len(positions),
        inference_separation=validate_separation(candidates[-1], inference, config["physics"], config["domain"]),
        interpretation="Reference validation only, not a recovery experiment. Errors are conditional Richardson estimates, not certified bounds.")
    write_json(output/"reference_summary.json", summary)
    return summary, resolved


if __name__ == "__main__":
    # Worker entry point deliberately performs no plotting/training imports.
    run_case(yaml.safe_load(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
