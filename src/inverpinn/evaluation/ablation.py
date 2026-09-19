"""Paired loss ablations, with no changes to the training/data protocol."""

import copy
import itertools

import numpy as np
import pandas as pd


def variants(base, pde_weights):
    """Change exactly one loss weight per condition; retain diagnostic losses.

    B retains IC/BC constraints but removes the only link from the unknown
    source to observations. Its source stays at initialization, not a learned
    localization. Zero-weight terms are still computed so random sample streams
    and diagnostic logging stay identical across paired runs.
    """
    weights = base["loss_weights"]
    if set(weights) != {"data", "pde", "initial", "boundary"} or any(
            not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Full model requires four positive finite weights.")
    if (not pde_weights or any(not np.isfinite(w) or w <= 0 for w in pde_weights)
            or len(set(pde_weights)) != len(pde_weights) or weights["pde"] in pde_weights):
        raise ValueError("PDE weights must be distinct, positive, and different from full model.")
    changes = {"A_full": {}, "B_no_pde": {"pde": 0.},
               "C_no_boundary": {"boundary": 0.}, "D_no_initial": {"initial": 0.}}
    changes.update({f"E_pde_{w:g}": {"pde": float(w)} for w in pde_weights})
    output = {}
    for name, change in changes.items():
        config = copy.deepcopy(base)
        config["loss_weights"].update(change)
        output[name] = config
    return output


def summarize(raw, datasets, seeds, conditions):
    """Within each dataset, mean/sample SD over paired training seeds.

    Also summarize paired error differences from A, without pooling datasets
    as if they were independent seeds. Reject failures/missing/duplicate runs.
    """
    keys = ["dataset_id", "seed", "condition"]
    expected = set(itertools.product(datasets, seeds, conditions))
    if (raw.duplicated(keys).any() or set(map(tuple, raw[keys].to_numpy())) != expected
            or not raw.status.eq("success").all()):
        raise ValueError("Need every paired run exactly once, all successful.")
    metrics = ["localization_error", "concentration_relative_l2"]
    if not np.isfinite(raw[metrics].to_numpy(dtype=float)).all() or (raw[metrics] < 0).any().any():
        raise ValueError("Errors must be finite and nonnegative.")
    rows = []
    for dataset in datasets:
        reference = raw[(raw.dataset_id == dataset) & (raw.condition == "A_full")].set_index("seed")
        for condition in conditions:
            group = raw[(raw.dataset_id == dataset) & (raw.condition == condition)].set_index("seed")
            row = dict(dataset_id=dataset, condition=condition, n_runs=len(group))
            for metric in metrics:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
                row[f"{metric}_paired_delta_mean"] = (group[metric]-reference[metric]).mean()
            rows.append(row)
    return pd.DataFrame(rows)
