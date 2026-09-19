"""Validate the existing finite-difference stencil; never regenerate datasets."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number
from inverpinn.physics.manufactured import error_norms, spatial_terms, time_factors

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def timestep(final_time, requested_dt):
    """Round the step count up: never exceed requested dt or change final time."""
    if not math.isfinite(final_time) or not math.isfinite(requested_dt) or min(final_time, requested_dt) <= 0:
        raise ValueError("final_time and requested_dt must be finite and positive.")
    steps = math.ceil(final_time/requested_dt)
    return final_time/steps, steps


def run_case(n, requested_dt, final_time, transport, *, profile=None, sources=()):
    """Execute the production solver, retaining only its final field.

    Precompute analytic spatial factors once on this exact grid. The callback
    computes S(t_n) = a'(t_n)P + a(t_n)A; it is bound to this run's grid.
    There is no alternative/reference stencil hidden in the validation runner.
    """
    start = time.perf_counter()
    dt, steps = timestep(final_time, requested_dt)
    axis = torch.linspace(0, 1, n, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    forcing, exact = None, None
    if profile is not None:
        p, a = spatial_terms(xx, yy, profile=profile, **transport)

        def forcing(_x, _y, t):
            value, derivative = time_factors(t, profile=profile)
            return derivative*p + value*a

        exact = time_factors(final_time, profile=profile)[0]*p
    numerical = solve_advection_diffusion(nx=n, ny=n, dt=dt, steps=steps,
        **transport, sources=sources, forcing=forcing, return_history=False)
    if not torch.isfinite(numerical).all():
        raise FloatingPointError("Non-finite numerical validation field.")
    row = dict(grid_size=n, dx=1/(n-1), dy=1/(n-1), dt=dt, steps=steps,
               final_time=dt*steps, **transport, runtime=time.perf_counter()-start,
               CFL_number=stability_number(dx=1/(n-1), dy=1/(n-1), dt=dt, **transport),
               empirical_order=None, Linf_empirical_order=None)
    if exact is not None:
        row.update(error_norms(numerical, exact, dx=row["dx"], dy=row["dy"]))
    return row, numerical, exact


def add_orders(rows, coordinate):
    """Compare adjacent errors, not a fitted or assumed theoretical slope."""
    for previous, current in zip(rows, rows[1:]):
        ratio = previous[coordinate]/current[coordinate]
        for norm, column in [("L2_error", "empirical_order"), ("Linf_error", "Linf_empirical_order")]:
            if min(previous[norm], current[norm]) > 0:
                current[column] = math.log(previous[norm]/current[norm])/math.log(ratio)


def restrict(field, n):
    """Extract coincident nodes only; fail rather than silently interpolate."""
    if (field.shape[0]-1) % (n-1):
        raise ValueError("Gaussian comparison grids must be nested.")
    stride = (field.shape[0]-1)//(n-1)
    if stride < 1:
        raise ValueError("Cannot restrict a coarser field onto a finer grid.")
    return field[::stride, ::stride]


def plot_convergence(rows, output, stem, coordinate, studies):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, label in studies:
        subset = [r for r in rows if r["study_type"] == name]
        for ax, norm in zip(axes, ["L2_error", "Linf_error"]):
            ax.loglog([r[coordinate] for r in subset], [r[norm] for r in subset], "o-", label=label)
            ax.set_xlabel("h (domain length units)" if coordinate == "dx" else "dt (synthetic time units)")
            ax.set_ylabel("Integral L2 (concentration × length)" if norm == "L2_error" else "Linf (concentration units)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
    fig.suptitle("Measured spatial convergence" if coordinate == "dx" else "Measured Euler time convergence; spatial truncation = 0")
    fig.savefig(output/f"{stem}.png")
    fig.savefig(output/f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def validate(config_path, output_dir=None):
    """Save one immutable validation run, including dirty-source provenance."""
    config = yaml.safe_load(Path(config_path).read_text())
    synthetic_path = (ROOT/config["synthetic_config"]).resolve()
    synthetic = yaml.safe_load(synthetic_path.read_text())
    solver = synthetic["solver"]
    output = Path(output_dir or ROOT/config["output_dir"]).resolve()
    # Fail before writing anything if previous results would be replaced.
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validation run: {output}")
    if solver["nx"] != solver["ny"]:
        raise ValueError("This refinement study requires a square synthetic grid.")
    grids = config["gaussian"]["grids"]
    if len(grids) < 3 or solver["nx"] not in grids or grids != sorted(set(grids)):
        raise ValueError("Use >=3 ascending unique Gaussian grids including the current grid.")
    if any((b-1) != 2*(a-1) for a, b in zip(grids, grids[1:])):
        raise ValueError("Richardson comparison requires successively doubled interval counts.")
    if config["temporal"]["u"] != 0 or config["temporal"]["v"] != 0:
        raise ValueError("Temporal isolation requires zero wind for spatially exact polynomial diffusion.")
    if config["spatial"]["time_check_factor"] != 2 or config["gaussian"]["time_check_factor"] != 2:
        raise ValueError("The dt/2 validation controls require time_check_factor=2.")
    output.mkdir(parents=True, exist_ok=False)
    resolved = config | dict(output_dir=str(output), synthetic_configuration=synthetic,
        dtype="float64", device="cpu", boundary="homogeneous Dirichlet on all edges", initial="zero")
    (output/"config.yaml").write_text(yaml.safe_dump(resolved))
    write_json(output/"environment.json", dict(python=platform.python_version(), platform=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        torch_threads=config["torch_threads"], deterministic_algorithms=True))
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source_files = [Path(__file__).resolve(), ROOT/"src/inverpinn/physics/finite_difference.py",
        ROOT/"src/inverpinn/physics/manufactured.py", ROOT/"src/inverpinn/physics/sources.py"]
    with zipfile.ZipFile(output/"source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(ROOT)))
    write_json(output/"provenance.json", dict(git_commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"), working_tree_status=git("status", "--short"),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        source_archive="source_snapshot.zip", synthetic_config_sha256=hashlib.sha256(synthetic_path.read_bytes()).hexdigest()))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({"savefig.dpi": config["plot_dpi"], "pdf.fonttype": 42, "font.size": 10})
    rows, fields, time_checks = [], {}, []
    spatial = config["spatial"]
    for study in ["spatial_advection_diffusion", "spatial_diffusion", "spatial_dt_half"]:
        transport = {k: spatial[k] for k in ("u", "v", "D")}
        if study == "spatial_diffusion":
            transport.update(u=0.0, v=0.0)
        subset = []
        for n in spatial["grids"]:
            dt = spatial["dt_over_h_squared"]/(n-1)**2
            if study == "spatial_dt_half":
                dt /= spatial["time_check_factor"]
            row, field, exact = run_case(n, dt, spatial["final_time"], transport, profile="sine")
            subset.append(row | dict(study_type=study, reference="analytic sine"))
            fields[f"{study}_{n}"] = field.numpy()
            fields[f"exact_sine_{n}"] = exact.numpy()
            print(f"{study}: {n}^2, dt={row['dt']:.6g}, L2={row['L2_error']:.6g}", flush=True)
        add_orders(subset, "dx")
        rows.extend(subset)
    temporal = config["temporal"]
    subset = []
    for level in range(temporal["levels"]):
        row, field, exact = run_case(temporal["grid"], temporal["initial_dt"]/2**level,
            temporal["final_time"], {k: temporal[k] for k in ("u", "v", "D")}, profile="polynomial")
        subset.append(row | dict(study_type="temporal", reference="analytic polynomial"))
        fields[f"temporal_{level}"] = field.numpy()
        fields["exact_temporal"] = exact.numpy()
    add_orders(subset, "dt")
    rows.extend(subset)

    gaussian_fields, gaussian_rows = {}, {}
    transport = {k: solver[k] for k in ("u", "v", "D")}
    final_time = solver["dt"]*solver["steps"]
    for n in grids:
        dt = solver["dt"]*((solver["nx"]-1)/(n-1))**2
        row, field, _ = run_case(n, dt, final_time, transport, sources=solver["sources"])
        gaussian_fields[n], gaussian_rows[n] = field, row
        fields[f"gaussian_{n}"] = field.numpy()
        print(f"gaussian: {n}^2, dt={row['dt']:.6g}, runtime={row['runtime']:.2f}s", flush=True)
    differences = []
    for coarse, fine in zip(grids, grids[1:]):
        row = gaussian_rows[coarse] | error_norms(gaussian_fields[coarse], restrict(gaussian_fields[fine], coarse),
            dx=1/(coarse-1), dy=1/(coarse-1))
        differences.append(row | dict(study_type="gaussian_grid_difference", reference=f"{fine}x{fine} numerical", reference_grid_size=fine))
    add_orders(differences, "dx")
    rows.extend(differences)
    # Include the finest grid's runtime/CFL without calling it an exact solution.
    rows.append(gaussian_rows[grids[-1]] | dict(study_type="gaussian_finest", reference="none; not exact"))
    for n in sorted({solver["nx"], grids[-1]}):
        row, field, _ = run_case(n, gaussian_rows[n]["dt"]/config["gaussian"]["time_check_factor"],
            final_time, transport, sources=solver["sources"])
        norms = error_norms(field, gaussian_fields[n], dx=1/(n-1), dy=1/(n-1))
        rows.append(row | norms | dict(study_type="gaussian_time_difference", reference="same grid, original dt", reference_dt=gaussian_rows[n]["dt"]))
        time_checks.append(dict(grid_size=n, dt=row["dt"], reference_dt=gaussian_rows[n]["dt"], **norms))
        fields[f"gaussian_dt_half_{n}"] = field.numpy()
    observed_p = differences[-1]["empirical_order"]
    if observed_p is None or observed_p <= 0:
        raise ValueError("Gaussian differences did not decrease: Richardson estimate is unjustified.")
    n = solver["nx"]
    coarse, fine = grids[-2:]
    reference_n = min(n, coarse)
    f = restrict(gaussian_fields[fine], reference_n)
    c = restrict(gaussian_fields[coarse], reference_n)
    extrapolated = f+(f-c)/(2**observed_p-1)
    current_error = error_norms(restrict(gaussian_fields[n], reference_n), extrapolated,
                               dx=1/(reference_n-1), dy=1/(reference_n-1))
    finest_error = error_norms(f, extrapolated, dx=1/(reference_n-1), dy=1/(reference_n-1))
    fields[f"gaussian_richardson_on_{reference_n}"] = extrapolated.numpy()
    spatial_time_ratios = []
    for n in spatial["grids"]:
        base = next(r for r in rows if r["study_type"] == "spatial_advection_diffusion" and r["grid_size"] == n)
        half = next(r for r in rows if r["study_type"] == "spatial_dt_half" and r["grid_size"] == n)
        spatial_time_ratios.append(dict(grid_size=n, relative_change_in_L2_error=abs(base["L2_error"]-half["L2_error"])/half["L2_error"]))
    target = config["gaussian"]["provisional_relative_L2_target"]
    summary = dict(
        scope="Numerical verification only; no dataset regeneration or neural training",
        units=dict(length="synthetic domain unit", time="synthetic time unit", concentration="synthetic concentration unit", runtime="seconds"),
        norm="L2 uses trapezoidal spatial integral; relative L2 is dimensionless; final time only",
        spatial_orders=[r["empirical_order"] for r in rows if r["study_type"] == "spatial_advection_diffusion" and r["empirical_order"] is not None],
        diffusion_orders=[r["empirical_order"] for r in rows if r["study_type"] == "spatial_diffusion" and r["empirical_order"] is not None],
        temporal_orders=[r["empirical_order"] for r in rows if r["study_type"] == "temporal" and r["empirical_order"] is not None],
        spatial_time_checks=spatial_time_ratios, gaussian_time_checks=time_checks,
        gaussian_difference_orders=[r["empirical_order"] for r in differences if r["empirical_order"] is not None],
        current_configuration=solver, current_CFL=gaussian_rows[solver["nx"]]["CFL_number"],
        current_is_stable=gaussian_rows[solver["nx"]]["CFL_number"] <= 1,
        current_error_Richardson_estimate=current_error, finest_error_Richardson_estimate=finest_error,
        provisional_relative_L2_target=target, target_is_project_requirement=False,
        current_meets_provisional_target=current_error["relative_L2_error"] <= target,
        finest_meets_provisional_target=finest_error["relative_L2_error"] <= target,
        finest_tested_configuration=dict(nx=fine, ny=fine, dx=1/(fine-1), dt=gaussian_rows[fine]["dt"], steps=gaussian_rows[fine]["steps"]),
        regenerate_recommendation="Regenerate accuracy-sensitive synthetic benchmarks after selecting and verifying a tolerance; do not treat current fields as exact truth.",
        caveats=["Richardson errors assume an asymptotic power law; these are estimates, not exact errors or confidence intervals.",
                 "Accuracy is assessed at final time; not a full space-time or sensor-weighted bound.",
                 "Validation is for the stated zero Dirichlet domain, not physical open-air boundary adequacy.",
                 "A one-percent target is provisional; no project-wide scientific tolerance was specified."])
    with (output/"numerical_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output/"final_fields.npz", **fields)
    plot_convergence(rows, output, "spatial_convergence", "dx", [
        ("spatial_advection_diffusion", "Wind + diffusion"), ("spatial_dt_half", "Wind + diffusion, dt/2"),
        ("spatial_diffusion", "Diffusion only")])
    plot_convergence(rows, output, "temporal_convergence", "dt", [("temporal", "Spatially exact polynomial")])
    write_json(output/"validation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/numerical_validation.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    validate(args.config, args.output_dir)


if __name__ == "__main__":
    main()
