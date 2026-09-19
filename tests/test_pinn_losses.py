"""Independent analytical MSE checks and weighted gradient/logging tests."""

import logging
from pathlib import Path

import pytest
import torch
import yaml

from inverpinn.training.losses import (
    data_loss, pde_loss, initial_condition_loss, boundary_condition_loss,
    combine_losses, log_losses,
)
from inverpinn.physics.autodiff import autograd_pde_residual


@pytest.mark.parametrize("loss_fn", [data_loss, initial_condition_loss, boundary_condition_loss])
def test_concentration_loss_exact_values_and_gradients(loss_fn):
    """Errors [1,-3] give MSE 5 and gradient [1,-3], independently per loss."""
    prediction = torch.tensor([[3.], [1.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[2.], [4.]], dtype=torch.float64)
    loss = loss_fn(prediction, target)
    assert loss.item() == 5
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[1.], [-3.]], dtype=torch.float64))
    assert loss_fn(target, target).item() == 0


def test_pde_loss_exact_values_and_gradients():
    residual = torch.tensor([1., -2., 3.], dtype=torch.float64, requires_grad=True)
    loss = pde_loss(residual)
    assert loss.item() == pytest.approx(14/3)
    loss.backward()
    torch.testing.assert_close(residual.grad, 2*residual.detach()/3)
    assert pde_loss(torch.zeros(4)).item() == 0


def test_pde_loss_retains_second_derivative_parameter_graph():
    points = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    C = a*(points[:, 0]**2 + points[:, 1]**2)
    residual = autograd_pde_residual(C, points, u=0., v=0., D=0.25, S=0.)
    loss = pde_loss(residual)  # residual=-a, hence loss=a² and derivative=2a.
    loss.backward()
    assert loss.item() == 4
    assert a.grad.item() == 4


def components():
    return {name: torch.tensor(float(i+1), requires_grad=True)
            for i, name in enumerate(("data", "pde", "initial", "boundary"))}


def test_weighted_total_and_gradients_from_configuration():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pinn.yaml").read_text())
    losses = components()
    assert combine_losses(losses, config["loss_weights"]).item() == 10
    weights = dict(data=2., pde=0., initial=0.5, boundary=3.)
    total = combine_losses(losses, weights)
    assert total.item() == 15.5
    total.backward()
    for key in losses:
        assert losses[key].grad.item() == weights[key]


def test_logging_includes_raw_weights_contributions_and_total(caplog):
    losses = components()
    weights = dict(data=2., pde=0., initial=0.5, boundary=3.)
    with caplog.at_level(logging.INFO):
        record = log_losses(logging.getLogger("inverpinn.test"), 12, losses, weights)
    assert record["step"] == 12 and record["loss/total"] == 15.5
    for name, loss in losses.items():
        assert record[f"loss/{name}"] == loss.item()
        assert record[f"weight/{name}"] == weights[name]
        assert record[f"weighted/{name}"] == weights[name]*loss.item()
        assert f"loss/{name}=" in caplog.text
    assert "loss/total=15.5" in caplog.text
    combine_losses(losses, weights).backward()  # logging must not destroy the graph.


@pytest.mark.parametrize("loss_fn", [data_loss, initial_condition_loss, boundary_condition_loss])
def test_rejects_accidental_broadcasting(loss_fn):
    with pytest.raises(ValueError, match="identical shapes"):
        loss_fn(torch.ones(3, 1), torch.ones(3))


@pytest.mark.parametrize("bad", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        pde_loss(bad)
    for loss_fn in (data_loss, initial_condition_loss, boundary_condition_loss):
        with pytest.raises(ValueError):
            loss_fn(bad, bad)


@pytest.mark.parametrize("weights", [dict(data=1), dict(data=0, pde=0, initial=0, boundary=0),
    dict(data=1, pde=-1, initial=1, boundary=1), dict(data=1, pde=float("nan"), initial=1, boundary=1)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        combine_losses(components(), weights)
