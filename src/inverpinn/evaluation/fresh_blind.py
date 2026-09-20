"""Preregistered paired analysis; no training, model selection or rescue.

Missing fits remain in denominators and tables. Continuous summaries use
explicit available-case counts; bootstrap uncertainty is conditional on the
fixed sensor layout/physics, not real-world uncertainty or posterior inference.
"""

import numpy as np
import pandas as pd

from inverpinn.data.fresh_blind import FRESH_IDS
from inverpinn.evaluation.single_source_benchmark import distribution, wilson_interval

PRIMARY = ("localization_error", "relative_strength_error", "relative_l2")
PAIRED = (*PRIMARY, "rmse", "mae", "pde_rmse")
METRICS = (*PAIRED, "pde_mae", "bc_rmse", "bc_maxabs", "ic_rmse", "ic_maxabs",
           "negative_fraction", "minimum_concentration", "training_seconds")


def primary_rows(rows, config, model):
    """Exactly one original-seed row per case; never replace it with a restart."""
    result = rows[(rows.model == model) & (rows.optimization_seed == config["optimization"]["primary_seed"])].copy()
    if len(result) != 30 or set(result.scenario_id) != set(FRESH_IDS):
        raise ValueError("All 30 unique primary rows, including failures, are required.")
    return result.set_index("scenario_id").loc[list(FRESH_IDS)].reset_index()


def compare_values(baseline, revised, atol, rtol):
    """Negative revised-minus-baseline differences favor revised; NaN is missing."""
    if baseline is None or revised is None or not np.isfinite([baseline, revised]).all():
        return None, "missing"
    difference = float(revised-baseline)
    tolerance = atol+rtol*max(abs(baseline), abs(revised))
    return difference, "tied" if abs(difference) <= tolerance else ("B_revised_better" if difference < 0 else "B0_better")


def bootstrap_difference(differences, seed, samples):
    """Paired percentile 95% CIs of mean and median differences; fixed RNG.

    Resample complete paired scenario indices with replacement. This is not a
    confidence interval for the finite list itself, nor a test of causation.
    """
    values = np.asarray([v for v in differences if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return dict(complete_pairs=0, mean=None, median=None, mean_CI95=None, median_CI95=None)
    draws = np.random.default_rng(seed).integers(len(values), size=(samples, len(values)))
    resampled = values[draws]
    return dict(complete_pairs=len(values), mean=float(values.mean()), median=float(np.median(values)),
        mean_CI95=np.quantile(resampled.mean(axis=1), [.025,.975]).tolist(),
        median_CI95=np.quantile(np.median(resampled, axis=1), [.025,.975]).tolist())


def paired_analysis(baseline, revised, settings):
    """Pair by ID and identical input/reference hashes, not by row order."""
    if baseline.scenario_id.duplicated().any() or revised.scenario_id.duplicated().any() or set(baseline.scenario_id) != set(revised.scenario_id):
        raise ValueError("Paired analysis requires identical unique scenario IDs.")
    left, right = baseline.set_index("scenario_id"), revised.set_index("scenario_id")
    records = []
    for identifier in sorted(left.index):
        a, b = left.loc[identifier], right.loc[identifier]
        for key in ("observations_hash", "reference_data_hash"):
            if pd.isna(a[key]) or a[key] != b[key]:
                raise ValueError(f"Paired cases do not share identical {key}.")
        row = dict(scenario_id=identifier, observations_hash=a.observations_hash, reference_data_hash=a.reference_data_hash)
        for key in PAIRED:
            difference, winner = compare_values(a[key], b[key], settings["metric_tie_atol"], settings["metric_tie_rtol"])
            row.update({f"B0_{key}":a[key], f"B_revised_{key}":b[key], f"delta_{key}":difference, f"comparison_{key}":winner})
        records.append(row)
    table = pd.DataFrame(records)
    summary = {}
    for key in PAIRED:
        counts = table[f"comparison_{key}"].value_counts()
        summary[key] = bootstrap_difference(table[f"delta_{key}"], settings["bootstrap_seed"], settings["bootstrap_samples"]) | {
            name:int(counts.get(name, 0)) for name in ("B_revised_better", "B0_better", "tied", "missing")}
    return table, summary


def seed_sensitivity(rows, config):
    """All three seeds, no best-seed selection; primary analysis stays separate."""
    seeds = {config["optimization"]["primary_seed"], *config["optimization"]["additional_seeds"]}
    subset = rows[(rows.model == "B_revised") & rows.scenario_id.isin(config["optimization"]["additional_seed_scenarios"])].copy()
    records = []
    for identifier in config["optimization"]["additional_seed_scenarios"]:
        group = subset[subset.scenario_id == identifier]
        if len(group) != 3 or set(group.optimization_seed) != seeds:
            raise ValueError("Seed substudy requires exactly three predeclared seeds per selected case.")
        row = dict(scenario_id=identifier, seeds=sorted(seeds), recovered=int(group.recovery_success.sum()),
                   completed=int(group.evaluation_success.sum()))
        for key in ("localization_error", "Q_pred", "relative_l2"):
            finite = group[key].dropna().to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            row.update({f"{key}_range":float(np.ptp(finite)) if len(finite) else None,
                        f"{key}_std":float(np.std(finite, ddof=1)) if len(finite)>1 else None,
                        f"{key}_count":len(finite)})
        records.append(row)
    return subset.sort_values(["scenario_id", "optimization_seed"]), records


def summarize_model(rows, config):
    recovered = int(rows.recovery_success.sum())
    settings = config["analysis"]
    signed = rows.Q_pred-rows.Q_true
    tol = settings["Q_equality_atol"]+settings["Q_equality_rtol"]*rows.Q_true.abs()
    result = dict(total=len(rows), computational_successes=int(rows.computational_success.sum()),
        evaluations=int(rows.evaluation_success.sum()), recovered=recovered, recovery_rate=recovered/len(rows),
        recovery_Wilson95=wilson_interval(recovered, len(rows), config["recovery"]["confidence_level"]),
        reference_target_misses=int((~rows.reference_target_met).sum()),
        metrics={k:distribution(rows[k].tolist(), len(rows)) for k in METRICS},
        Q_bias=dict(under=int((signed < -tol).sum()), over=int((signed > tol).sum()),
                    approximately_equal=int((signed.abs() <= tol).sum()), missing=int(signed.isna().sum()),
                    signed_error=distribution(signed.tolist(), len(rows)),
                    signed_relative_error=distribution((signed/rows.Q_true).tolist(), len(rows))))
    return result


def interpretation(baseline, revised, config):
    """Apply the prospective reporting rubric, not an outcome-fitted score."""
    rules = config["analysis"]["classification"]
    b, r = baseline["metrics"], revised["metrics"]
    physical = (revised["evaluations"] == 30 and revised["computational_successes"] == 30
                and all(r[k]["max"] == 0 for k in ("negative_fraction", "bc_maxabs", "ic_maxabs")))
    comparable = baseline["evaluations"] == 30 and physical and r["pde_rmse"]["mean"] <= b["pde_rmse"]["mean"]
    if comparable and revised["recovered"] >= rules["strong_min_recovered"] and r["relative_l2"]["median"] <= rules["strong_max_median_L2"]:
        return "strong controlled recovery"
    if comparable and revised["recovered"] >= rules["conditional_min_recovered"]:
        nonregression = all(r[k][s] <= b[k][s] for k in PRIMARY for s in ("mean", "median"))
        improvement = all(b[k]["median"] > 0 and 1-r[k]["median"]/b[k]["median"] >= rules["conditional_min_relative_median_improvement"]
                          for k in ("relative_strength_error", "relative_l2"))
        if nonregression and improvement:
            return "conditional controlled recovery"
    return "unreliable controlled recovery"


def profile_strength(unit_readings, observations):
    r"""Q_hat=max(0,<g,y>/<g,g>) minimizes mean((Q g-y)^2), Q>=0.

    Zero is an admissible limiting strength, not a clipped PINN parameter.
    No source truth or upper bound is an argument. Same-grid linearity applies
    because transport, interpolation, source width and homogeneous IC/BC are
    fixed. A different reference grid leaves numerical-model discrepancy.
    """
    g, y = np.asarray(unit_readings, dtype=float), np.asarray(observations, dtype=float)
    if g.shape != y.shape or g.size == 0 or not np.isfinite(g).all() or not np.isfinite(y).all():
        raise ValueError("Matched finite nonempty sensor trajectories required.")
    square = float(np.sum(g*g))
    if square <= 0:
        raise ValueError("Zero source response has undefined profiled strength.")
    q = max(0., float(np.sum(g*y)/square))
    return q, float(np.mean((q*g-y)**2))


def exploratory_correlations(rows, features):
    """Descriptive Pearson/Spearman associations, no causal or tuning role."""
    merged = rows.merge(features, on="scenario_id", validate="one_to_one")
    records = []
    for key in PRIMARY:
        for feature in ("x_true", "y_true", "Q_true", "nearest_sensor_distance", "boundary_distance", "observed_sensor_rms"):
            valid = merged[[key, feature]].dropna()
            for method in ("pearson", "spearman"):
                corr = valid[key].corr(valid[feature], method=method) if len(valid)>1 and valid[key].nunique()>1 and valid[feature].nunique()>1 else np.nan
                records.append(dict(outcome=key, feature=feature, method=method, count=len(valid),
                                    correlation=float(corr) if np.isfinite(corr) else None, exploratory=True))
    return records
