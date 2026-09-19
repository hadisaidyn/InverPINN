"""Strict sparse-only interface to the unchanged inverse PINN optimizer.

This module has no data-file/evaluation imports. Its fit function cannot receive
scenario paths, metadata, source truth or dense reference grids. This is a
dataflow restriction, not an OS sandbox against intentional filesystem access.
"""

from dataclasses import dataclass
import logging
import math

import torch

from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.training.forward_pinn import train_forward_pinn


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Allowlisted optimizer choices; no dataset/ground-truth fields exist."""

    width: int
    depth: int
    steps: int
    learning_rate: float
    data_batch: int
    collocation_batch: int
    initial_batch: int
    boundary_per_edge: int
    data_weight: float
    pde_weight: float
    initial_weight: float
    boundary_weight: float
    initial_x: float
    initial_y: float
    initial_Q: float

    @classmethod
    def from_mapping(cls, values):
        """Reject unrecognized fields rather than forwarding a broad config."""
        values = dict(values)
        weights = values.pop("loss_weights")
        if set(weights) != {"data", "pde", "initial", "boundary"}:
            raise ValueError("Exactly four loss weights are required.")
        return cls(**values, **{f"{k}_weight": float(v) for k, v in weights.items()})

    def optimizer_config(self, seed):
        """Construct just the keys used by the existing optimizer."""
        names = ("width", "depth", "steps", "learning_rate", "data_batch",
                 "collocation_batch", "initial_batch", "boundary_per_edge")
        return {name: getattr(self, name) for name in names} | dict(
            experiment={"seed": seed}, loss_weights={
                name: getattr(self, f"{name}_weight")
                for name in ("data", "pde", "initial", "boundary")})


@dataclass(frozen=True, slots=True)
class SparseTrainingData:
    """Only C readings[T,N], positions[N,2], times[T] and known u,v,D.

    Tensor copies sever backing-array references to a loader; coordinates use
    the unit square and zero-start physical time. No arbitrary physics mapping
    is retained. The experiment wrapper verifies known zero IC/Dirichlet BC.
    """

    positions: torch.Tensor
    times: torch.Tensor
    measurements: torch.Tensor
    u: float
    v: float
    D: float

    def __post_init__(self):
        for name in ("positions", "times", "measurements"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError("Sparse inputs must be finite tensors.")
            object.__setattr__(self, name, tensor.detach().clone().contiguous())
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("Positions must be [N,2].")
        if self.measurements.shape != (len(self.times), len(self.positions)):
            raise ValueError("Measurements must be [T,N].")
        if not all(type(getattr(self, k)) in (int, float) and math.isfinite(getattr(self, k)) for k in ("u", "v", "D")):
            raise ValueError("Transport must contain finite numeric scalars only.")


def fit_sparse_source(data: SparseTrainingData, settings: TrainingSettings,
                      sigma: float, optimization_seed: int, logger=None, on_step=None):
    """Jointly fit (MLP, xs,ys,Q) using sparse tensors and known physics only.

    Sigma is a known source-family assumption, never estimated using truth.
    The fixed source guess is independent of all observations/source scenarios;
    the seed controls MLP initialization and collocation/data sampling only.
    """
    if type(data) is not SparseTrainingData or type(settings) is not TrainingSettings:
        raise TypeError("Use the strict sparse data and training-settings records.")
    source = TrainableGaussianSource(settings.initial_x, settings.initial_y, settings.initial_Q, sigma)
    model, optimizer, history = train_forward_pinn(
        data.positions, data.times, data.measurements,
        {k: getattr(data, k) for k in ("u", "v", "D")},
        settings.optimizer_config(optimization_seed), logger or logging.getLogger(__name__),
        on_step=on_step, source_model=source)
    return model, source, optimizer, history
