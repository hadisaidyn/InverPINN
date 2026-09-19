"""End-to-end reproduction and numerical-truth checks on the unified pipeline."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from inverpinn.data.sensors import sample_sensors


def test_pipeline_reproducible_and_self_contained(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/dataset.yaml").read_text())
    config["solver"].update(nx=11, ny=13, steps=4)
    config["solver"]["sources"].append(dict(x_s=0.7, y_s=0.3, Q=0.5, sigma=0.1))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    command = [sys.executable, str(root / "scripts/generate_synthetic_dataset.py"), "--config", str(path)]
    for name in ("first", "second"):
        subprocess.run(command + ["--output-dir", str(tmp_path / name)], check=True, capture_output=True, text=True)
    output = tmp_path / "first"
    with np.load(output / "dataset.npz", allow_pickle=False) as a, np.load(tmp_path / "second/dataset.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert a["C"].shape == (5, 13, 11)
        np.testing.assert_array_equal(a["source_positions"], [[0.3, 0.7], [0.7, 0.3]])
        np.testing.assert_array_equal(a["source_Q"], [1., 0.5])
        assert a["seed"].item() == config["seed"]
        for key in ("u", "v", "D"):
            assert a[key].item() == config["solver"][key]
        for n in (5, 10, 20, 50):
            assert a[f"measurements_{n}"].shape == (5, n)
            _, clean = sample_sensors(torch.from_numpy(a["C"]), torch.from_numpy(a["x"]),
                torch.from_numpy(a["y"]), positions=torch.from_numpy(a[f"sensor_positions_{n}"]))
            np.testing.assert_array_equal(clean.numpy(), a[f"clean_measurements_{n}"])
        metadata = json.loads((output / "metadata.json").read_text())
        assert set(metadata["array_dimensions"]) == set(a.files)
    assert metadata["config"] == yaml.safe_load((output / "config.yaml").read_text())
    assert metadata["dataset_sha256"] == hashlib.sha256((output / "dataset.npz").read_bytes()).hexdigest()
    assert (output / "metrics.json").is_file()
    with Image.open(output / "sensor_positions.png") as image:
        image.verify()
    # Re-running into the same directory must preserve the completed experiment.
    result = subprocess.run(command + ["--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode != 0 and "Output already exists" in result.stderr
    assert metadata["dataset_sha256"] == hashlib.sha256((output / "dataset.npz").read_bytes()).hexdigest()
