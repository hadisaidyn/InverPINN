"""Exercise the visualization command and its reproducible saved artifacts."""

import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from PIL import Image


def test_visualization_saves_consistent_artifacts(tmp_path: Path) -> None:
    """The plot, metrics, field checkpoint, and config describe the same run."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/synthetic.yaml").read_text())
    config["output_dir"] = str(tmp_path / "results")
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config))
    subprocess.run(
        [sys.executable, str(root / "scripts/visualize_source.py"), "--config", str(config_path)],
        check=True, cwd=root, capture_output=True, text=True,
    )
    output = Path(config["output_dir"])
    with Image.open(output / "synthetic_source.png") as image:
        image.verify()
    saved_config = yaml.safe_load((output / "synthetic_source_config.yaml").read_text())
    assert saved_config == config
    checkpoint = torch.load(output / "synthetic_source.pt", weights_only=True)
    assert checkpoint["config"] == config
    metrics = json.loads((output / "synthetic_source_metrics.json").read_text())
    field = checkpoint["source_intensity"]
    assert field.shape == (config["grid"]["points"],) * 2
    assert metrics["sampled_max"] == field.max().item()
    assert metrics["analytic_peak"] == config["source"]["Q"]
