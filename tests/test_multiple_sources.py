"""Mixture gradients and optimal permutation-aware evaluation."""

import itertools
import logging

import numpy as np
import pytest
import torch

from inverpinn.physics.trainable_source import TrainableGaussianMixture
from inverpinn.evaluation.metrics import matched_source_metrics
from inverpinn.training.forward_pinn import train_forward_pinn

SPECS = [dict(x_s=0.3,y_s=0.7,Q=1.,sigma=0.08), dict(x_s=0.7,y_s=0.3,Q=0.8,sigma=0.08)]


def test_mixture_sum_and_permutation():
    model = TrainableGaussianMixture(SPECS)
    reversed_model = TrainableGaussianMixture(list(reversed(SPECS)))
    x, y = torch.tensor([0.31,0.72]), torch.tensor([0.69,0.28])
    torch.testing.assert_close(model(x,y), sum(s(x,y) for s in model.sources))
    torch.testing.assert_close(model(x,y), reversed_model(x,y))
    model(x,y).sum().backward()
    assert len(list(model.parameters())) == 6
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad) and p.grad.abs() > 0


def test_swapped_sources_have_zero_errors():
    result = matched_source_metrics([[0.7,0.3],[0.3,0.7]], [0.8,1.], [[0.3,0.7],[0.7,0.3]], [1.,0.8])
    assert result["mean_localization_error"] == 0
    assert result["mean_relative_strength_error"] == 0
    assert [p["true_index"] for p in result["pairs"]] == [1,0]


def test_matching_is_global_not_greedy():
    p = np.array([[0.4,0.],[0.1,0.]])
    t = np.array([[0.,0.],[1.,0.]])
    result = matched_source_metrics(p, [2.,1.], t, [1.,2.])
    assert result["mean_localization_error"] == pytest.approx(0.35)
    assert result["mean_relative_strength_error"] == 0
    optimal = min(sum(np.linalg.norm(p[i]-t[j]) for i,j in enumerate(order)) for order in itertools.permutations(range(2)))
    assert sum(pair["localization_error"] for pair in result["pairs"]) == pytest.approx(optimal)


def test_unmatched_counts_rejected():
    with pytest.raises(ValueError):
        matched_source_metrics([[0.,0.]], [1.], [[0.,0.],[1.,1.]], [1.,1.])


def test_two_sources_joint_training_and_checkpoint():
    source = TrainableGaussianMixture(SPECS)
    initial = {k:v.clone() for k,v in source.state_dict().items()}
    config = dict(experiment=dict(seed=3), steps=2, width=8, depth=2, learning_rate=0.001,
        data_batch=4, collocation_batch=64, initial_batch=4, boundary_per_edge=2,
        loss_weights=dict(data=1.,pde=1.,initial=1.,boundary=1.))
    _, optimizer, history = train_forward_pinn(torch.tensor([[0.3,0.7],[0.7,0.3]]), torch.tensor([0.,0.1]),
        torch.tensor([[0.,0.],[0.1,0.08]]), dict(u=0.1,v=0.,D=0.01), config, logging.getLogger("test.multi"), source_model=source)
    for i in range(2):
        assert f"source/{i}/Q" in history[-1]
        assert not torch.equal(source.state_dict()[f"sources.{i}.raw_Q"], initial[f"sources.{i}.raw_Q"])
    restored = TrainableGaussianMixture(SPECS)
    restored.load_state_dict(source.state_dict())
    assert restored.estimates() == source.estimates()
