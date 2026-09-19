"""Reproduce configured paper experiments without overwriting previous runs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "dataset": "generate_synthetic_dataset.py", "data_baseline": "train_data_baseline.py",
    "forward_pinn": "train_forward_pinn.py", "inverse_pinn": "train_inverse_pinn.py",
    "evaluation": "evaluate_inverse_pinn.py", "sensor_sweep": "run_sensor_sweep.py",
    "noise_sweep": "run_noise_sweep.py", "wind_sweep": "run_wind_sweep.py",
    "ablation": "run_ablation.py", "spatial_baselines": "evaluate_spatial_baselines.py",
    "classical_inverse": "fit_classical_inverse.py", "real_case_study": "run_real_case_study.py",
    "inversions": "diagnose_inversions.py",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def merge(base, override):
    """Recursively overlay explicit settings, without changing source configs."""
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolve(value, previous):
    """Resolve ${stage_id} paths only against earlier enabled stages."""
    if isinstance(value, dict):
        return {k:resolve(v, previous) for k,v in value.items()}
    if isinstance(value, list):
        return [resolve(v, previous) for v in value]
    if isinstance(value, str):
        def replacement(match):
            if match[1] not in previous:
                raise ValueError(f"Unknown, disabled or forward dependency: {match[1]}")
            return str(previous[match[1]])
        return re.sub(r"\$\{([^}]+)\}", replacement, value)
    return value


def git_info(root):
    """Record HEAD and dirty state honestly, including a non-Git workspace."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode:
        return dict(commit=None, status="unavailable", reason=head.stderr.strip(), dirty=None)
    status = git("status", "--porcelain", "--untracked-files=all")
    return dict(commit=head.stdout.strip(), status="available", dirty=bool(status.stdout.strip()),
                working_tree_status=status.stdout, diff=git("diff", "HEAD", "--binary").stdout)


def prepare(config, output):
    """Validate stage IDs/order and freeze nested configuration dependencies."""
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("A master seed in [0, 2**32) is required.")
    if config.get("publication", {}).get("dpi", 0) < 300:
        raise ValueError("Publication PNG resolution must be at least 300 dpi.")
    previous, seen, stages, snapshots = {}, set(), [], {}
    def freeze(value, stack=()):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key, item in value.items():
            if key.endswith("_config") and isinstance(item, str):
                path = (ROOT/item).resolve()
                if path in stack:
                    raise ValueError("Cyclic configuration dependency")
                frozen = freeze(yaml.safe_load(path.read_text()), stack+(path,))
                target = output/"configs"/"dependencies"/(sha256(path)+".yaml")
                snapshots[target] = frozen
                result[key] = str(target)
        return result
    for stage in config["stages"]:
        name = stage["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen:
            raise ValueError(f"Invalid or duplicate stage ID: {name}")
        seen.add(name)
        if stage["kind"] not in SCRIPTS:
            raise ValueError(f"Unknown experiment kind: {stage['kind']}")
        if not stage.get("enabled", True):
            if not stage.get("reason"):
                raise ValueError("Disabled stages need an explicit reason.")
            stages.append(stage | dict(status="excluded"))
            continue
        base = yaml.safe_load((ROOT/stage["config"]).read_text()) if stage.get("config") else {}
        settings = freeze(resolve(merge(base, stage.get("overrides", {})), previous))
        target = output/"experiments"/name
        settings["output_dir"] = str(target)
        if not stage.get("required"):
            raise ValueError(f"Stage {name} needs an explicit required-artifact list.")
        for artifact in stage["required"]:
            p = Path(artifact)
            if p.is_absolute() or ".." in p.parts or not p.parts:
                raise ValueError("Required artifacts must be relative paths inside the experiment.")
        previous[name] = target
        stages.append(stage | dict(status="pending", settings=settings))
    if not previous:
        raise ValueError("At least one enabled stage is required.")
    return stages, snapshots


def worker(kind, config_path, seed, dpi):
    """Isolated deterministic execution; render original artists to PDF and PNG.

    No image upscaling: PNG uses the requested dpi and PDF retains vector text
    and lines. Descendant training processes retain their original diagnostic
    plots; the selected top-level paper figures use this publication style.
    """
    import random
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    matplotlib.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.bbox":"tight"})
    original = Figure.savefig
    def save(figure, filename, *args, **kwargs):
        path = Path(filename)
        if path.suffix.lower() == ".png":
            kwargs["dpi"] = dpi
            original(figure, filename, *args, **kwargs)
            pdf_kwargs = {k:v for k,v in kwargs.items() if k not in ("format", "metadata")}
            original(figure, path.with_suffix(".pdf"), *args, **pdf_kwargs,
                     metadata={"CreationDate":None, "ModDate":None})
        else:
            original(figure, filename, *args, **kwargs)
    Figure.savefig = save
    script = ROOT/"scripts"/SCRIPTS[kind]
    if kind == "evaluation":
        settings = yaml.safe_load(Path(config_path).read_text())
        namespace = runpy.run_path(str(script))
        namespace["evaluate_run"](Path(settings["run_dir"]), Path(settings["output_dir"]))
        Path(settings["output_dir"], "config.yaml").write_text(yaml.safe_dump(settings))
    else:
        sys.argv = [str(script), "--config", str(config_path)]
        runpy.run_path(str(script), run_name="__main__")


def collect(output, stages):
    """Assemble selected stage-level figures/tables and hash every original artifact."""
    import csv
    rows = []
    for stage in stages:
        directory = output/"experiments"/stage["id"]
        if stage["status"] != "complete":
            continue
        for source in sorted(directory.iterdir()):
            category = "figures" if source.suffix in (".pdf", ".png") else "tables" if (
                source.suffix == ".csv" and "history" not in source.stem or
                source.name in ("metrics.json", "summary.json")) else None
            if category and source.is_file():
                target = output/"paper"/category/(stage["id"]+"__"+source.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for source in sorted(directory.rglob("*")):
            if source.is_file():
                rows.append(dict(stage=stage["id"], path=str(source.relative_to(output)),
                                 bytes=source.stat().st_size, sha256=sha256(source)))
    with (output/"artifacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/"artifacts.json", rows)


def reproduce(config_path, output_override=None):
    """Run in a fresh exclusive directory; failures preserve logs and halt the suite."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    output = Path(output_override or ROOT/config["output_dir"]).resolve()
    stages, snapshots = prepare(config, output)
    git = git_info(ROOT)
    if config.get("require_git_commit", False) and git["commit"] is None:
        raise ValueError("A Git commit is required by this paper configuration.")
    output.mkdir(parents=True, exist_ok=False)  # Atomic reservation; no force/overwrite mode.
    (output/"configs").mkdir()
    (output/"experiments").mkdir()
    (output/"logs").mkdir()
    (output/"config.yaml").write_text(yaml.safe_dump(config | dict(output_dir=str(output))))
    shutil.copy2(config_path, output/"configs"/"paper_source.yaml")
    for target, values in snapshots.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(values))
    env_settings = dict(PYTHONHASHSEED=str(config["seed"]), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        MPLBACKEND="Agg")
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        machine=platform.machine(), packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        process_settings=env_settings, determinism="CPU, one thread, PyTorch deterministic algorithms; bitwise identity across platforms/versions not promised")
    write_json(output/"environment.json", environment)
    write_json(output/"git.json", git)
    sources = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
               *ROOT.joinpath("configs").glob("*.yaml"), ROOT/"requirements.txt", ROOT/"pyproject.toml"]
    hashes = {}
    for source in sources:
        destination = output/"source_snapshot"/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(source.relative_to(ROOT))] = sha256(source)
    write_json(output/"source_hashes.json", hashes)
    state = dict(status="running", label=config["label"], git_commit=git["commit"], stages=stages)
    write_json(output/"run.json", state)
    try:
        for stage in stages:
            if stage["status"] == "excluded":
                continue
            stage["status"] = "running"
            write_json(output/"run.json", state)
            path = output/"configs"/(stage["id"]+".yaml")
            path.write_text(yaml.safe_dump(stage["settings"]))
            print(f"Running {stage['id']} ({config['label']})", flush=True)
            with (output/"logs"/(stage["id"]+".log")).open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", stage["kind"],
                    "--config", str(path), "--seed", str(config["seed"]),
                    "--dpi", str(config["publication"]["dpi"])], cwd=ROOT, env=os.environ | env_settings,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = output/"experiments"/stage["id"]
            for artifact in ["config.yaml", *stage["required"]]:
                if not (directory/artifact).is_file():
                    raise RuntimeError(f"Required artifact missing: {stage['id']}/{artifact}")
            # Includes nested sweep training runs: provenance beside every config.
            for saved in directory.rglob("config.yaml"):
                write_json(saved.parent/"paper_provenance.json", dict(git=git,
                    environment=environment, master_seed=config["seed"], source_hashes=hashes,
                    stage=stage["id"], label=config["label"], config_sha256=sha256(saved)))
            stage["status"] = "complete"
            write_json(output/"run.json", state)
        collect(output, stages)
        state["status"] = "complete"
        (output/"README.md").write_text(f"# {config['label']}\n\n"
            "Selected top-level figures (300+ dpi PNG and vector PDF) and tables are in paper/.\n"
            "All metrics, checkpoints and nested diagnostic plots remain in experiments/.\n"
            "run.json records exclusions; excluded real-data stages are not reproduced claims.\n"
            "artifacts.csv/json index original files with hashes. environment.json and git.json record provenance.\n"
            "Do not interpret a smoke run as paper-quality scientific evidence.\n")
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for stage in stages:
            if stage["status"] == "running":
                stage["status"] = "failed"
        raise
    finally:
        write_json(output/"run.json", state)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/paper.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", choices=SCRIPTS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--dpi", type=int, default=300, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.config, args.seed, args.dpi)
    else:
        print(reproduce(args.config, args.output_dir))
