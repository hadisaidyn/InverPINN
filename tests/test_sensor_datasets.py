"""Verify all requested dataset sizes and their generated artifacts."""

from pathlib import Path
import subprocess
import sys

from PIL import Image
import pandas as pd
import torch
import yaml


def test_sensor_dataset_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "truth.pt"
    axis = torch.linspace(0, 1, 3, dtype=torch.float64)
    torch.save({"C": torch.ones((2, 3, 3), dtype=torch.float64), "x": axis,
                "y": axis, "t": torch.tensor([0., 1.]), "config": {}}, checkpoint)
    config = yaml.safe_load((root / "configs/sensors.yaml").read_text())
    config.update(forward_checkpoint=str(checkpoint), dataset_dir=str(tmp_path / "data"),
                  output_dir=str(tmp_path / "plots"))
    config_path = tmp_path / "sensors.yaml"
    config_path.write_text(yaml.safe_dump(config))
    subprocess.run([sys.executable, str(root / "scripts/generate_sensor_datasets.py"),
                    "--config", str(config_path)], check=True, capture_output=True, text=True)
    for n in (5, 10, 20, 50):
        saved = torch.load(tmp_path / "data" / f"sensors_{n}.pt", weights_only=True)
        assert saved["measurements"].shape == (2, n)
        assert saved["positions"].shape == (n, 2)
        assert "C" not in saved
        table = pd.read_csv(tmp_path / "data" / f"sensors_{n}.csv")
        assert len(table) == 2*n
        torch.testing.assert_close(torch.tensor(table.concentration.values).reshape(2, n), saved["measurements"])
        assert (tmp_path / "plots" / f"sensors_{n}_metrics.json").is_file()
        assert saved["config"] == yaml.safe_load((tmp_path / "plots" / f"sensors_{n}_config.yaml").read_text())
        with Image.open(tmp_path / "plots" / f"sensors_{n}.png") as image:
            image.verify()
    with Image.open(tmp_path / "plots" / "sensor_positions.png") as image:
        image.verify()
