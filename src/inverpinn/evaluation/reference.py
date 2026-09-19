"""Inter-grid differences and conditional Richardson estimates, never exact errors."""

import math

import torch

from inverpinn.physics.manufactured import error_norms


def compare_references(cases):
    """Compare three successively doubled grids at identical physical outputs.

    For each selected field time, restrict all three fields to the coarsest
    nodes (no spatial interpolation), and measure d0=||C_h-C_h/2|| and
    d1=||C_h/2-C_h/4||. Under the *assumed* asymptotic expansion C_h=C*+k h^p,
    p=log2(d0/d1), and C*=C_fine+(C_fine-C_medium)/(2^p-1).
    Report all three errors relative to this extrapolation. They are estimates,
    not bounds. Non-decreasing/zero differences cannot identify p: report null.

    Sensor trajectories are compared at the exact same physical coordinates
    and measurement times, with a pooled vector L2 norm (not per-reading
    relative errors near zero). Field L2 uses trapezoidal spatial integration.
    Only selected times are assessed, not an unsampled space-time supremum.
    """
    if len(cases) != 3:
        raise ValueError("Richardson comparison requires exactly three grids.")
    sizes = [c["fields"].shape[-1] for c in cases]
    if any(c["fields"].shape[-2] != n for c, n in zip(cases, sizes)) or any(
            b-1 != 2*(a-1) for a, b in zip(sizes, sizes[1:])):
        raise ValueError("Require square grids with doubled interval counts.")
    for key in ("field_times", "measurement_times", "sensor_positions"):
        if any(not torch.equal(cases[0][key], c[key]) for c in cases[1:]):
            raise ValueError(f"Reference comparisons require identical {key}.")
    h = 1/(sizes[0]-1)

    def estimate(values, norm, scope, time=None):
        coarse, medium, fine = values
        d0, d1 = norm(coarse-medium), norm(medium-fine)
        order = math.log2(d0/d1) if d0 > d1 > 0 else None
        extrapolated = fine+(fine-medium)/(2**order-1) if order is not None else None
        denominator = norm(extrapolated) if extrapolated is not None else 0
        rows = []
        for i, (n, value) in enumerate(zip(sizes, values)):
            next_norm = norm(values[i+1]) if i < 2 else 0
            rows.append(dict(scope=scope, time=time, grid_size=n, comparison_grid=sizes[0],
                next_grid_size=sizes[i+1] if i < 2 else None,
                intergrid_L2=(d0 if i == 0 else d1) if i < 2 else None,
                intergrid_relative_L2=norm(value-values[i+1])/next_norm if next_norm else None,
                Richardson_order=order,
                Richardson_relative_L2_estimate=norm(value-extrapolated)/denominator if denominator else None,
                error_kind="measured inter-grid difference and conditional Richardson estimate; no exact Gaussian error"))
        return rows

    rows = []
    for i, t in enumerate(cases[0]["field_times"].tolist()):
        fields = [c["fields"][i, ::2**level, ::2**level] for level, c in enumerate(cases)]
        norm = lambda value: error_norms(value, torch.zeros_like(value), dx=h, dy=h)["L2_error"]
        rows.extend(estimate(fields, norm, "selected_field", t))
    rows.extend(estimate([c["measurements"] for c in cases], lambda a: float(torch.linalg.vector_norm(a)), "sensor_trajectory"))
    # Final differences on each pair's native coarse grid, also independently
    # inspectable; these need not exactly match coarsest-grid quadrature above.
    for i in range(2):
        norms = error_norms(cases[i]["fields"][-1], cases[i+1]["fields"][-1, ::2, ::2],
                            dx=1/(sizes[i]-1), dy=1/(sizes[i]-1))
        rows.append(dict(scope="native_final_difference", time=float(cases[i]["field_times"][-1]),
            grid_size=sizes[i], next_grid_size=sizes[i+1], comparison_grid=sizes[i],
            intergrid_L2=norms["L2_error"], intergrid_Linf=norms["Linf_error"],
            intergrid_relative_L2=norms["relative_L2_error"], Richardson_order=None,
            Richardson_relative_L2_estimate=None, error_kind="measured inter-grid difference only"))
    return rows


def choose_reference(rows, runtimes, target):
    """Choose the cheapest measured candidate meeting all assessed estimates.

    Every requested field time and the pooled sensor trajectory must have a
    defined estimate <= target. A null estimate is NOT a zero error/pass.
    This decision is conditional on Richardson assumptions and sampled times.
    """
    if not 0 < target < 1:
        raise ValueError("Reference relative-error target must be in (0,1).")
    eligible, per_grid = [], {}
    for n, runtime in runtimes.items():
        estimates = [r["Richardson_relative_L2_estimate"] for r in rows if r["grid_size"] == n
                     and r["scope"] in ("selected_field", "sensor_trajectory")]
        defined = bool(estimates) and all(v is not None and math.isfinite(v) for v in estimates)
        maximum = max(estimates) if defined else None
        passed = defined and maximum <= target
        per_grid[str(n)] = dict(max_assessed_Richardson_relative_L2_estimate=maximum, meets_estimated_target=passed)
        if passed:
            eligible.append((runtime, n))
    return dict(selected_grid=min(eligible)[1] if eligible else None,
                target_relative_L2=target, per_grid=per_grid, certified_bound=False,
                scope="Selected field times and pooled fixed-sensor trajectory only")
