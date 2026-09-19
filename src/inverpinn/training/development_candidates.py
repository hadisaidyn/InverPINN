"""Small controlled revision-4 optimizer; receives sparse tensors, never truth.

Sampling and Adam ordering intentionally match the original trainer. The only
differences are explicit concentration transforms, Q transform, and loss-unit
factors. A bitwise regression test compares the identity configuration to B0.
No early stopping, best-epoch selection, adaptive sampling or L-BFGS is hidden.
"""

import logging

import torch

from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
from inverpinn.physics.nondimensional import CharacteristicScales
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.training.forward_pinn import require_finite, train_forward_pinn
from inverpinn.training.losses import combine_losses, log_losses
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings


def fit_audited_baseline(data, settings, sigma, seed, logger=None, on_step=None, on_objective=None):
    """Instrument lower-level training without broadening the frozen sparse API."""
    if type(data) is not SparseTrainingData or type(settings) is not TrainingSettings:
        raise TypeError("Use strict sparse data and settings.")
    source = TrainableGaussianSource(settings.initial_x, settings.initial_y, settings.initial_Q, sigma)
    model, optimizer, history = train_forward_pinn(data.positions, data.times, data.measurements,
        {k:getattr(data, k) for k in ("u", "v", "D")}, settings.optimizer_config(seed),
        logger or logging.getLogger(__name__), on_step=on_step, source_model=source, on_objective=on_objective)
    return model, source, optimizer, history


def loss_factors(formulation):
    if set(formulation) != {"constraint", "Q_parameterization", "scales"}:
        raise ValueError("Formulation must have exactly constraint, Q_parameterization, scales.")
    scales = formulation["scales"]
    return (dict(data=1., pde=1., initial=1., boundary=1.) if scales is None
            else CharacteristicScales(**scales).loss_factors())


def fit_candidate(data, settings, sigma, seed, formulation, logger=None, on_step=None, on_objective=None):
    """Minimize scaled physical losses using fixed-budget, deterministic Adam.

    Raw logged losses are in physical units. Effective weights = dimensionless
    preference weights times explicit unit factors. On hard-constrained models
    the IC/BC terms are exactly zero, retaining their differentiable graphs.
    """
    if type(data) is not SparseTrainingData or type(settings) is not TrainingSettings:
        raise TypeError("Use strict sparse data and settings.")
    factors = loss_factors(formulation)
    config = settings.optimizer_config(seed)
    weights = {k:config["loss_weights"][k]*factors[k] for k in factors}
    logger = logger or logging.getLogger(__name__)
    source = DiagnosticGaussianSource(settings.initial_x, settings.initial_y, settings.initial_Q,
                                      sigma, formulation["Q_parameterization"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed+1)
    upper = torch.tensor([1., 1., data.times[-1].item()])
    model = ConstrainedConcentrationMLP(data.times[-1].item(), settings.width, settings.depth,
                                       formulation["constraint"])
    parameters = list(model.parameters())+list(source.parameters())
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    points = torch.cat((data.positions.repeat(len(data.times), 1),
                        data.times.repeat_interleave(len(data.positions))[:, None]), 1).float()
    targets = data.measurements.reshape(-1, 1).float()
    history = []
    def sample(n):
        return torch.rand((n, 3), generator=generator)*upper
    for step in range(settings.steps):
        ids = torch.randint(len(points), (settings.data_batch,), generator=generator)
        interior = sample(settings.collocation_batch).requires_grad_()
        initial = sample(settings.initial_batch)
        initial[:, 2] = 0
        edges = []
        for axis, value in ((0, 0.), (0, 1.), (1, 0.), (1, 1.)):
            edge = sample(settings.boundary_per_edge)
            edge[:, axis] = value
            edges.append(edge)
        boundary = torch.cat(edges)
        S = source(interior[:, :1], interior[:, 1:2])
        optimizer.zero_grad()
        sensor_prediction, field = model(points[ids]), model(interior)
        ic, bc = model(initial), model(boundary)
        r = autograd_pde_residual(field, interior, u=data.u, v=data.v, D=data.D, S=S)
        losses = dict(data=(sensor_prediction-targets[ids]).square().mean(), pde=r.square().mean(),
                      initial=ic.square().mean(), boundary=bc.square().mean())
        total = combine_losses(losses, weights)
        require_finite(total, f"candidate step {step}: objective")
        if on_objective:
            on_objective(step, losses, weights, model, source, interior)
        total.backward()
        for p in parameters:
            if p.grad is not None:
                require_finite(p.grad, f"candidate step {step}: gradient")
        record = log_losses(logger, step, losses, weights)
        optimizer.step()
        for p in parameters:
            require_finite(p, f"candidate step {step}: parameter")
        estimate = source.estimates()
        if not all(torch.isfinite(torch.tensor(v)) for v in estimate.values()):
            raise FloatingPointError("Nonfinite transformed source parameter")
        record.update({f"source/{k}":v for k,v in estimate.items()})
        record["epoch"] = step+1
        history.append(record)
        if on_step:
            on_step(record)
    return model, source, optimizer, history


def select_multistart(objectives):
    """Only {start_name: scalar sparse/physics objective}; no truth/metric rows."""
    import math
    if not objectives or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in objectives.values()):
        raise ValueError("Finite nonnegative objective scalars are required.")
    return min(objectives, key=lambda name:(objectives[name], name))
