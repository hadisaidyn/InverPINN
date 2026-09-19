"""Exercise the forward command and its optional animation."""
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import torch
import yaml


def test_forward_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/forward.yaml").read_text())
    config["solver"].update(nx=11, ny=13, steps=3)
    config["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "forward.yaml"
    path.write_text(yaml.safe_dump(config))
    subprocess.run([sys.executable, str(root / "scripts/run_forward_simulation.py"),
                    "--config", str(path), "--animate"], check=True, capture_output=True, text=True)
    output = Path(config["output_dir"])
    for name in ("forward_simulation_final.png", "forward_simulation.gif"):
        with Image.open(output / name) as image:
            image.verify()
    saved = torch.load(output / "forward_simulation.pt", weights_only=True)
    assert saved["C"].shape == (4, 13, 11)
    assert saved["t"][-1].item() == 3 * config["solver"]["dt"]
    metrics = json.loads((output / "forward_simulation_metrics.json").read_text())
    assert metrics["final_maximum"] == saved["C"][-1].max().item()
    assert metrics["stability_number"] <= 1
    assert saved["config"] == yaml.safe_load((output / "forward_simulation_config.yaml").read_text())
