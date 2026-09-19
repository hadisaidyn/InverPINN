"""Revision-4 allowlist. Never enumerate the consumed benchmark's held-out set.

This is an accidental-access guard, not a security sandbox. Input loaders read
only sparse observations/known physics. Truth is accessible for evaluation
only after a terminal training record; all paths must stay inside one of the
ten existing development directories, including through symbolic links.
"""

import json
from pathlib import Path

from inverpinn.data.synthetic_reference import load_model_inputs, sha256

DEVELOPMENT_IDS = tuple(f"development_{i:03d}" for i in range(1, 11))


def development_path(root, scenario_id, relative=""):
    """Reject non-development identifiers before any scenario file is opened."""
    if scenario_id not in DEVELOPMENT_IDS:
        raise ValueError("Revision 4 permits development_001 through development_010 ONLY.")
    datasets = (Path(root)/"datasets").resolve()
    lexical = datasets/scenario_id
    case = lexical.resolve()
    if case != lexical or "heldout" in str(case).lower():
        raise ValueError("Development directory may not alias another dataset.")
    path = (case/relative).resolve()
    if not path.is_relative_to(case) or "heldout" in str(path).lower():
        raise ValueError("Path escapes the development allowlist.")
    return path


def sparse_inputs(root, scenario_id):
    """Validate both sparse paths before delegating to the strict input loader."""
    for name in ("inputs/observations.npz", "inputs/physics.json"):
        development_path(root, scenario_id, name)
    return load_model_inputs(development_path(root, scenario_id))


def development_truth(root, scenario_id, run):
    """Read only this case's truth, after a persisted terminal training barrier."""
    case = development_path(root, scenario_id)
    attempt = json.loads((Path(run)/"attempt.json").read_text())
    if attempt.get("training_terminated") is not True:
        raise RuntimeError("Truth access requires terminated training.")
    manifest = json.loads(development_path(root, scenario_id, "generation.json").read_text())
    if manifest["split"] != "development" or manifest["scenario_id"] != scenario_id:
        raise ValueError("Mislabeled development dataset.")
    reference = (Path(root)/manifest["reference_path"]).resolve()
    if not reference.is_relative_to(case):
        raise ValueError("Reference path escapes its development case.")
    reference = development_path(root, scenario_id, str(reference.relative_to(case)))
    if sha256(reference) != manifest["reference_data_hash"]:
        raise ValueError("Reference ground truth was modified.")
    source = json.loads(development_path(root, scenario_id, "truth/source.json").read_text())
    from inverpinn.data.single_source_benchmark import object_hash
    if object_hash(source) != manifest["source_hash"]:
        raise ValueError("Source ground truth was modified.")
    return source, reference, manifest


def development_hashes(root):
    """Hash every existing development file, without opening other scenarios."""
    hashes = {}
    for identifier in DEVELOPMENT_IDS:
        case = development_path(root, identifier)
        if not case.is_dir():
            raise FileNotFoundError(case)
        for path in sorted(case.rglob("*")):
            checked = development_path(root, identifier, str(path.relative_to(case)))
            if checked.is_file():
                hashes[str(path.relative_to(Path(root).resolve()))] = sha256(checked)
    return hashes


def validate_benchmark_open(root, path, mode="r", flags=0):
    """Deny shared manifests/held-out paths and ALL benchmark writes.

    Used by an optional process audit hook as a second guard. Only individual
    development dataset files (and old development-run diagnostics) may be
    opened underneath the old benchmark root. Source code outside it is not
    restricted. Integer file descriptors cannot identify a new path and are
    ignored; normal Python/numpy/torch pathname opens are covered.
    """
    import os
    if isinstance(path, int):
        return
    root, path = Path(root).resolve(), Path(os.fsdecode(path)).resolve()
    if not path.is_relative_to(root):
        return
    parts = path.relative_to(root).parts
    if len(parts) < 3 or parts[0] not in ("datasets", "runs") or parts[1] not in DEVELOPMENT_IDS:
        raise PermissionError("Revision-4 audit guard: consumed or shared benchmark path denied.")
    if any(c in (mode or "") for c in "wax+") or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        raise PermissionError("Revision-4 audit guard: original benchmark is read-only.")


def install_development_read_guard(root):
    """Install a process-lifetime Python open hook; not an OS security sandbox."""
    import sys
    root = Path(root).resolve()
    def guard(event, args):
        if event == "open":
            validate_benchmark_open(root, *args)
    sys.addaudithook(guard)
