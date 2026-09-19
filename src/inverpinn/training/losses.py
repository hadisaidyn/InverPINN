"""Modular PINN objectives and explicit, configuration-supplied weighting."""

import logging
import math
from collections.abc import Mapping

import torch


LOSS_NAMES = ("data", "pde", "initial", "boundary")


def _finite_nonempty(value: torch.Tensor) -> None:
    if not value.is_floating_point() or value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError("Loss inputs must be nonempty, finite floating-point tensors.")


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _finite_nonempty(prediction)
    _finite_nonempty(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes; broadcasting is not allowed.")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device.")
    return (prediction - target).square().mean()


def data_loss(prediction: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    r"""Mean sensor error: L_data = (1/N) sum_i (C_hat_i - C_observed_i)^2.

    Each reading is one equally weighted observation. Units are concentration².
    Only sensor observations belong here, not dense simulation truth.
    """
    return _mse(prediction, observations)


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    r"""L_pde = mean(r²), for r=C_t+u*C_x+v*C_y-D*(C_xx+C_yy)-S.

    Pass residuals evaluated at interior collocation points by the physics
    module. This function neither computes derivatives nor detaches their graph.
    Units are (concentration/time)², unlike the other three objectives.
    """
    _finite_nonempty(residual)
    return residual.square().mean()


def initial_condition_loss(prediction: torch.Tensor, initial_values: torch.Tensor) -> torch.Tensor:
    r"""L_initial = mean((C_hat(x,y,t0)-C0(x,y))²), in concentration².

    The caller evaluates the model at the initial time. Explicit target values
    allow nonzero initial conditions; zeros are not assumed by this function.
    """
    return _mse(prediction, initial_values)


def boundary_condition_loss(prediction: torch.Tensor, boundary_values: torch.Tensor) -> torch.Tensor:
    r"""Dirichlet loss L_boundary = mean((C_hat(edge,t)-g(edge,t))²).

    The caller samples boundary coordinates and supplies their target
    concentrations g. For our forward solver g=0 on all four edges. This
    implements concentration/Dirichlet constraints, not flux/Neumann conditions.
    All supplied points are equally weighted; sampling controls edge weighting.
    Units are concentration².
    """
    return _mse(prediction, boundary_values)


def combine_losses(losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    """Sum weight[k]*loss[k] for all four names, with no default weights.

    Pass config['loss_weights'] directly as weights. All four keys must be
    explicit. Weights are finite, nonnegative scalars and at least one must be
    positive. A zero weight disables a contribution but it remains available
    for logging. Each component must be a finite nonnegative scalar tensor.
    All tensors must share a device. Gradients flow through every component.

    Because PDE MSE has different units, weights encode both physical scaling
    and relative importance. Values are an experiment choice, not universal
    constants. No automatic normalization or adaptive weighting is applied.
    """
    if set(losses) != set(LOSS_NAMES) or set(weights) != set(LOSS_NAMES):
        raise ValueError(f"Losses and weights must have exactly these keys: {LOSS_NAMES}.")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in weights.values()) or not any(w > 0 for w in weights.values()):
        raise ValueError("Weights must be finite nonnegative numbers, with at least one positive.")
    for loss in losses.values():
        _finite_nonempty(loss)
        if loss.ndim != 0 or loss.item() < 0:
            raise ValueError("Each component loss must be a nonnegative scalar tensor.")
    if len({loss.device for loss in losses.values()}) != 1:
        raise ValueError("All component losses must share a device.")
    return sum(weights[name] * losses[name] for name in LOSS_NAMES)


def log_losses(logger: logging.Logger, step: int, losses: Mapping[str, torch.Tensor],
               weights: Mapping[str, float]) -> dict[str, int | float]:
    """Log and return a detached record suitable for CSV or JSONL history.

    Includes step, each raw loss, each weight, each weighted contribution, and
    the recomputed total. Call after computing losses at one common model state.
    Use a FileHandler to persist text logs; no global logger setup is performed.
    Logging leaves input tensors and their autograd graphs intact.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer.")
    total = combine_losses(losses, weights)
    record = {"step": step}
    for name in LOSS_NAMES:
        record[f"loss/{name}"] = losses[name].detach().item()
        record[f"weight/{name}"] = float(weights[name])
        record[f"weighted/{name}"] = (weights[name] * losses[name]).detach().item()
    record["loss/total"] = total.detach().item()
    logger.info("PINN losses %s", " ".join(f"{key}={value:.8g}" for key, value in record.items()))
    return record
