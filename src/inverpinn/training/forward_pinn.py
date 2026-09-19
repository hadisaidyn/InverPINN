"""Forward PINN optimization with fixed, known transport and emission parameters."""

import logging
import math

import torch

from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.physics.sources import gaussian_source
from inverpinn.training.losses import (data_loss, pde_loss, initial_condition_loss,
    boundary_condition_loss, combine_losses, log_losses)


def require_finite(value, name):
    """Abort immediately with diagnostic context on NaN or infinity."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"NaN/Inf detected in {name}")


def train_forward_pinn(positions, times, observations, known, config, logger, on_step=None, source_model=None,
                       on_objective=None):
    """Train using sensors and physics only; dense truth is not an argument.

    C starts unconstrained. Soft MSE terms impose zero initial concentration
    and zero Dirichlet values on all four edges of [0,1]². Interior, initial,
    and edge points are resampled uniformly each update. Known Gaussian sources
    and u,v,D are fixed values, never trainable parameters. Losses use physical
    coordinates/concentrations without target scaling. CPU float32 and local
    sampling seeds give reproducibility in the same software environment.
    For inverse use, source_model supplies a trainable Gaussian instead of
    known['sources']; its parameters join the same optimizer. No true source
    location or strength is then needed. Post-update estimates are logged each
    epoch (one optimizer step), alongside pre-update sampled loss values.
    """
    for name, value in (("positions", positions), ("times", times), ("observations", observations)):
        require_finite(value, name)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or not torch.all(times[1:] > times[:-1]):
        raise ValueError("Times must start at zero and increase strictly.")
    if positions.ndim != 2 or positions.shape[1] != 2 or observations.shape != (len(times), len(positions)):
        raise ValueError("Expected positions[N,2] and observations[T,N].")
    if not torch.all((positions >= 0) & (positions <= 1)):
        raise ValueError("Sensors must be inside the unit square.")
    for key in ("steps", "data_batch", "collocation_batch", "initial_batch", "boundary_per_edge"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    for key in ("u", "v", "D"):
        if not math.isfinite(known[key]):
            raise FloatingPointError(f"NaN/Inf detected in known {key}")
    if known["D"] < 0:
        raise ValueError("D must be nonnegative.")
    weights = config["loss_weights"]
    combine_losses({name: torch.tensor(0.) for name in ("data", "pde", "initial", "boundary")}, weights)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["experiment"]["seed"])
    generator = torch.Generator().manual_seed(config["experiment"]["seed"]+1)
    upper = [1., 1., times[-1].item()]
    model = ConcentrationMLP([0., 0., 0.], upper, config["width"], config["depth"])
    parameters = list(model.parameters())
    if source_model is not None:
        parameters += list(source_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    points = torch.cat([positions.repeat(len(times), 1), times.repeat_interleave(len(positions))[:, None]], 1).float()
    targets = observations.reshape(-1, 1).float()

    def sample(n):
        return torch.rand((n, 3), generator=generator)*torch.tensor(upper)

    history = []
    for step in range(config["steps"]):
        ids = torch.randint(len(points), (config["data_batch"],), generator=generator)
        interior = sample(config["collocation_batch"]).requires_grad_()
        initial = sample(config["initial_batch"])
        initial[:, 2] = 0
        edges = []
        for axis, value in ((0, 0.), (0, 1.), (1, 0.), (1, 1.)):
            edge = sample(config["boundary_per_edge"])
            edge[:, axis] = value
            edges.append(edge)
        boundary = torch.cat(edges)
        if source_model is None:
            source = torch.zeros((len(interior), 1))
            for s in known["sources"]:
                source = source + gaussian_source(interior[:, :1], interior[:, 1:2], **s)
        else:
            source = source_model(interior[:, :1], interior[:, 1:2])
        optimizer.zero_grad()
        sensor_prediction, field = model(points[ids]), model(interior)
        ic, bc = model(initial), model(boundary)
        for name, value in (("sensor predictions", sensor_prediction), ("interior predictions", field),
                            ("initial predictions", ic), ("boundary predictions", bc), ("source", source)):
            require_finite(value, f"step {step}: {name}")
        residual = autograd_pde_residual(field, interior, u=known["u"], v=known["v"], D=known["D"], S=source)
        require_finite(residual, f"step {step}: residual")
        losses = dict(data=data_loss(sensor_prediction, targets[ids]), pde=pde_loss(residual),
                      initial=initial_condition_loss(ic, torch.zeros_like(ic)),
                      boundary=boundary_condition_loss(bc, torch.zeros_like(bc)))
        for name, value in losses.items():
            require_finite(value, f"step {step}: {name} loss")
        total = combine_losses(losses, weights)
        require_finite(total, f"step {step}: total loss")
        # Optional read-only diagnostic hook. It must retain the graph and must
        # not consume RNG state, update parameters, or accumulate .grad values.
        if on_objective is not None:
            on_objective(step, losses, weights, model, source_model, interior)
        total.backward()
        named_parameters = list(model.named_parameters())
        if source_model is not None:
            named_parameters += [(f"source.{name}", p) for name, p in source_model.named_parameters()]
        for name, parameter in named_parameters:
            if parameter.grad is not None:
                require_finite(parameter.grad, f"step {step}: gradient {name}")
        record = log_losses(logger, step, losses, weights)
        optimizer.step()
        for name, parameter in named_parameters:
            require_finite(parameter, f"step {step}: parameter {name}")
        if source_model is not None:
            estimates = source_model.estimates()
            if "sources" in estimates:
                record.update({f"source/{i}/{key}": value for i, s in enumerate(estimates["sources"]) for key, value in s.items()})
                logger.info("Sources epoch=%d %s", step+1, estimates)
            else:
                record.update({f"source/{key}": value for key, value in estimates.items()})
                logger.info("Source epoch=%d x_s=%.9g y_s=%.9g Q=%.9g", step+1,
                            estimates["x_s"], estimates["y_s"], estimates["Q"])
            record["epoch"] = step+1
        history.append(record)
        if on_step:
            on_step(record)
        if (step+1) % 500 == 0:
            print(f"Step {step+1}: total={total.item():.6g}, PDE={losses['pde'].item():.6g}", flush=True)
    return model, optimizer, history
