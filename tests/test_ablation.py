"""Weight isolation, complete paired summaries and no-PDE identifiability."""

import copy
import logging
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from inverpinn.evaluation.ablation import variants, summarize
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.training.forward_pinn import train_forward_pinn


def config():
    return dict(experiment=dict(seed=3), steps=2, width=8, depth=2, learning_rate=0.001,
        data_batch=4, collocation_batch=16, initial_batch=4, boundary_per_edge=2,
        loss_weights=dict(data=10.,pde=1.,initial=1.,boundary=1.))


def test_only_requested_weight_changes():
    base = config()
    before = copy.deepcopy(base)
    cases = variants(base, [0.1,10.])
    assert base == before
    assert cases["A_full"] == base
    for name, key, value in [("B_no_pde","pde",0.), ("C_no_boundary","boundary",0.),
                             ("D_no_initial","initial",0.), ("E_pde_0.1","pde",0.1), ("E_pde_10","pde",10.)]:
        wanted = copy.deepcopy(base)
        wanted["loss_weights"][key] = value
        assert cases[name] == wanted


def test_no_pde_source_stays_fixed_and_initial_losses_are_paired():
    histories = []
    for name in ("A_full", "B_no_pde"):
        source = TrainableGaussianSource(x_s=0.5,y_s=0.5,Q=0.5,sigma=0.1)
        before = {k:v.clone() for k,v in source.state_dict().items()}
        _, _, history = train_forward_pinn(torch.tensor([[0.3,0.7]]), torch.tensor([0.,0.1]),
            torch.tensor([[0.],[0.1]]), dict(u=0.1,v=0.,D=0.01), variants(config(),[0.1])[name],
            logging.getLogger("test.ablation"), source_model=source)
        histories.append(history)
        if name == "B_no_pde":
            for key in before:
                assert torch.equal(source.state_dict()[key], before[key])
    for loss in ("data","pde","initial","boundary"):
        assert histories[0][0][f"loss/{loss}"] == histories[1][0][f"loss/{loss}"]


def test_paired_summary_and_failures():
    raw = pd.DataFrame([dict(dataset_id=0,seed=s,condition=c,status="success",
        localization_error=s+offset,concentration_relative_l2=s+offset)
        for s in (0,1) for c,offset in (("A_full",0),("B_no_pde",2))])
    summary = summarize(raw,[0],[0,1],["A_full","B_no_pde"])
    assert summary.iloc[1].localization_error_paired_delta_mean == 2
    assert summary.iloc[0].localization_error_std == pytest.approx(np.sqrt(0.5))
    for bad in (raw.iloc[:-1],pd.concat([raw,raw.iloc[:1]]),raw.assign(status="failed")):
        with pytest.raises(ValueError, match="paired run"):
            summarize(bad,[0],[0,1],["A_full","B_no_pde"])


@pytest.mark.parametrize("weights", [[1.], [0.], [0.1,0.1], [float("nan")]])
def test_invalid_sweep_weights(weights):
    with pytest.raises(ValueError):
        variants(config(), weights)


def test_runner_pairs_configurations_and_preserves_dotted_log_names(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ablation_runner",
        Path(__file__).resolve().parents[1] / "scripts/run_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = tmp_path / "data.npz"
    np.savez(dataset, sensor_positions_1=[[0.3,0.7]], measurements_1=[[0.],[0.1]])
    base = config() | dict(n_sensors=1)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    def fake_training(command, **kwargs):
        settings = yaml.safe_load(Path(command[-1]).read_text())
        folder = Path(settings["output_dir"])
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(dict(relative_l2=0.1,
            matched_sources=dict(mean_localization_error=0.01))))
        kwargs["stdout"].write(folder.name)

    monkeypatch.setattr(module.subprocess, "run", fake_training)
    output = tmp_path / "ablation"
    settings = dict(inverse_config=str(base_path), datasets=[str(dataset)], seeds=[0,1],
                    pde_weights=[0.1,10.], workers=2, output_dir=str(output))
    summary = module.run(settings)
    assert len(summary) == 6
    assert len(list((output / "runs").glob("*.log"))) == 12
    for seed in (0,1):
        assert (output / "runs" / f"dataset_0_E_pde_0.1_seed_{seed}.log").exists()
    for path in (output / "configs").glob("*.yaml"):
        saved = yaml.safe_load(path.read_text())
        assert saved["dataset"] == str(dataset.resolve())
        assert saved["steps"] == base["steps"]
        assert saved["experiment"]["seed"] in (0,1)
    for name in ("config.yaml","raw_results.csv","summary.csv","report.md","ablation_comparison.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        module.run(settings)
