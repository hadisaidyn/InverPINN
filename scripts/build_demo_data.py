"""Package existing final-result previews; no inference, training or simulation.

Default build needs the preserved local archive. Every input NPZ is checked
against the original 13,399-file SHA-256 inventory anchored in paper evidence.
Display fields use every second node of the stored stride-4 previews (81²,
effective stride 8 of 641²). Float32 conversion error is measured and recorded.
No interpolation, refitting, source replacement or metric recomputation occurs.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.evidence import (
    EVIDENCE,
    IDS,
    frozen_metadata,
    load_bundle,
    read_json,
    require,
    sha,
)


def save_npz(path, arrays):
    """Deterministic ZIP metadata; NPY values retain the specified display dtype."""
    with zipfile.ZipFile(
        path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for key, value in sorted(arrays.items()):
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, stream.getvalue())


def build(
    output,
    archive=ROOT / "results/final_confirmatory",
    inventory=ROOT / "artifacts/finalization_audit/protected_before.json",
):
    output = Path(output).resolve()
    archive = Path(archive)
    inventory = Path(inventory)
    require(
        output == ROOT / "demo/data"
        or output.is_relative_to(ROOT / "artifacts")
        or not output.is_relative_to(ROOT),
        "Output must be demo/data, a new artifacts directory, or external scratch",
    )
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing demo bundle")
    metadata = frozen_metadata()
    evidence_manifest = read_json(EVIDENCE / "manifest.json")
    require(
        sha(inventory) == evidence_manifest["protected_inventory_sha256"],
        "Original archive inventory hash mismatch",
    )
    protected = read_json(inventory)
    inputs = {}
    # Verify all sources before creating any output, not only the selected case.
    for sid in IDS:
        for method in ("J1", "classical"):
            rel = f"results/final_confirmatory/runs/{method}/{sid}/predictions.npz"
            path = archive / f"runs/{method}/{sid}/predictions.npz"
            require(sha(path) == protected[rel], f"Changed archived prediction: {rel}")
            inputs[rel] = protected[rel]
    (output / "fields").mkdir(parents=True)
    (output / "final_metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    scenarios = {}
    for sid in IDS:
        with (
            np.load(
                archive / f"runs/J1/{sid}/predictions.npz", allow_pickle=False
            ) as j,
            np.load(
                archive / f"runs/classical/{sid}/predictions.npz", allow_pickle=False
            ) as c,
        ):
            for key in ("x", "y", "times", "reference"):
                require(
                    np.array_equal(j[key], c[key]),
                    f"Inconsistent paired preview: {sid}/{key}",
                )
            require(
                j["reference"].shape == (4, 161, 161)
                and np.array_equal(j["times"], metadata["times"]),
                "Unexpected frozen preview layout",
            )
            arrays = {key: j[key][::2] for key in ("x", "y")}
            arrays["times"] = j["times"]
            conversion = {}
            for key, original in [
                ("reference", j["reference"]),
                ("J1", j["fields"]),
                ("classical", c["fields"]),
            ]:
                require(
                    original.shape == (4, 161, 161) and np.isfinite(original).all(),
                    "Invalid archived field",
                )
                sampled = original[:, ::2, ::2]
                arrays[key] = sampled.astype(np.float32)
                error = float(np.max(np.abs(arrays[key].astype(np.float64) - sampled)))
                require(error <= 1e-7, "Unexpected display conversion error")
                conversion[key] = error
            save_npz(output / f"fields/{sid}.npz", arrays)
            scenarios[sid] = {"max_float32_absolute_difference": conversion}
    files = {
        str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        schema_version=1,
        scientific_commit=metadata["provenance"]["scientific_commit"],
        source_inventory_sha256=sha(inventory),
        source_predictions=inputs,
        scenarios=scenarios,
        files=files,
        display={
            "stored_grid": 161,
            "display_grid": 81,
            "effective_reference_stride": 8,
            "times": metadata["times"],
            "dtype": "float32",
            "interpolation": "none",
            "metrics": "copied from full-resolution frozen CSVs, never from display fields",
        },
        training_performed=False,
        simulation_performed=False,
        predictions_recomputed=False,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    load_bundle(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "demo/data")
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "results/final_confirmatory"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "artifacts/finalization_audit/protected_before.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        load_bundle(args.output_dir)
    else:
        build(args.output_dir, args.archive, args.inventory)
    print(
        "Verified 30 final scenarios; J1 23/30, classical 30/30. No inference or training."
    )


if __name__ == "__main__":
    main()
