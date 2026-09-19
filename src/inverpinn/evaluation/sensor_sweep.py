"""Statistics for complete, paired sensor-count experiments."""

import numpy as np
import pandas as pd


ERROR_COLUMNS = ("localization_error", "relative_strength_error", "concentration_relative_l2")


def summarize_results(raw: pd.DataFrame, counts: list[int], seeds: list[int]) -> pd.DataFrame:
    """Compute arithmetic means and sample std (ddof=1) across seeds.

    Each sensor count must contain exactly the requested seeds, with no failed
    or missing runs and no nonfinite metrics. No outliers are excluded. These
    error bars describe seed variability, not confidence intervals. Relative
    errors are fractions, and localization uses domain length units.
    """
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or len(set(counts)) != len(counts):
        raise ValueError("Need distinct counts and at least two distinct seeds.")
    expected = {(n, seed) for n in counts for seed in seeds}
    if len(raw) != len(expected) or set(zip(raw["n_sensors"], raw["seed"])) != expected:
        raise ValueError("Results must contain exactly one row per sensor count and seed.")
    if not raw["status"].eq("success").all():
        raise ValueError("Cannot summarize failed runs as a complete experiment.")
    values = raw[list(ERROR_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Errors must be finite and nonnegative; undefined metrics cannot be summarized.")
    rows = []
    for count in counts:
        group = raw[raw["n_sensors"] == count]
        row = {"n_sensors": count, "n_runs": len(group)}
        for name in ERROR_COLUMNS:
            row[f"{name}_mean"] = group[name].mean()
            row[f"{name}_std"] = group[name].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)
