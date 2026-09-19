"""Evaluate saved inverse PINN artifacts without retraining or changing them."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from inverpinn.evaluation.metrics import concentration_metrics, matched_source_metrics


def evaluate_run(run_dir: Path, output: Path) -> dict:
    """Write JSON/Markdown report and optimally matched source overlay.

    Verify reference provenance and coordinate equality before comparing arrays.
    Source errors use the final estimated parameters, not a best-truth checkpoint.
    The map overlays both locations on final numerical concentration.
    """
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dataset = Path(config["dataset"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Reference dataset checksum does not match training provenance.")
    estimate = json.loads((run_dir / "source_estimate.json").read_text())
    estimates = estimate.get("sources", [estimate])
    with np.load(dataset, allow_pickle=False) as data, np.load(run_dir / "predictions.npz", allow_pickle=False) as pred:
        for coordinate in ("x", "y", "t"):
            if not np.array_equal(data[coordinate], pred[coordinate]):
                raise ValueError(f"Prediction/reference {coordinate} coordinates do not match.")
        positions, strengths = data["source_positions"], data["source_Q"]
        matching = matched_source_metrics([[s["x_s"], s["y_s"]] for s in estimates],
            [s["Q"] for s in estimates], positions, strengths)
        truth, predicted = data["C"], pred["C_predicted"]
        x, y, t = data["x"], data["y"], data["t"]
        if truth.shape != (len(t), len(y), len(x)):
            raise ValueError("Reference must have shape [time,y,x].")
        full = concentration_metrics(predicted, truth)
        final = concentration_metrics(predicted[-1], truth[-1])
        true_sources = [dict(x_s=float(p[0]), y_s=float(p[1]), Q=float(q)) for p,q in zip(positions,strengths)]
        true_source = true_sources[0] if len(true_sources) == 1 else {"sources": true_sources}
        location = matching["mean_localization_error"]
        strength = matching["mean_relative_strength_error"]
        final_field = truth[-1].copy()
        x, y, t = x.copy(), y.copy(), t.copy()
    report = {"true_source": true_source, "predicted_source": estimate,
              "source_localization_error": location, "relative_source_strength_error": strength,
              "space_time": full, "final_time": final,
              "matched_sources": matching,
              "scope": "Spatially matched final source estimates; full grid including sensors; no held-out simulation",
              "reference": "Numerical finite-difference reference, not exact analytical concentration",
              "units": {"localization": "domain length", "strength_relative_error": "fraction",
                        "relative_l2": "fraction", "mae": "concentration", "rmse": "concentration"},
              "run_dir": str(run_dir), "dataset": str(dataset), "dataset_sha256": config["dataset_sha256"]}
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def number(value, percent=False):
        return "undefined (zero reference)" if value is None else (f"{100*value:.3f}%" if percent else f"{value:.8g}")

    source_table = ["| Predicted index → true index | True (x,y,Q) | Predicted (x,y,Q) | Location error |",
                    "|---|---|---|---:|"]
    for pair in matching["pairs"]:
        i,j = pair["predicted_index"], pair["true_index"]
        actual, predicted_source = true_sources[j], estimates[i]
        source_table.append(f"| {i} → {j} | {actual['x_s']:.6f}, {actual['y_s']:.6f}, {actual['Q']:.6f} | {predicted_source['x_s']:.6f}, {predicted_source['y_s']:.6f}, {predicted_source['Q']:.6f} | {pair['localization_error']:.6f} |")
    source_table_text = "\n".join(source_table)

    text = f"""# Inverse PINN evaluation

Final {len(estimates)}-source estimate after {config.get('steps', 'configured')} optimizer steps.

{source_table_text}

- Mean matched Euclidean localization error: **{number(location)} domain units**.
- Mean matched relative source-strength error: **{number(strength, True)}**.

| Concentration metric | Full space-time grid | Final time |
|---|---:|---:|
| Relative L2 | {number(full['relative_l2'], True)} | {number(final['relative_l2'], True)} |
| MAE | {number(full['mae'])} | {number(final['mae'])} |
| RMSE | {number(full['rmse'])} | {number(final['rmse'])} |

Localization is sqrt((x_pred-x_true)^2+(y_pred-y_true)^2).
Strength error is abs(Q_pred-Q_true)/abs(Q_true).
Concentration relative L2 is norm(C_pred-C_true)/norm(C_true).
MAE and RMSE use all supplied entries with equal weight and concentration units.
Relative errors with a zero denominator are undefined (null in JSON).

The reference is a numerical finite-difference simulation. These reconstruction
metrics include sensor locations and do not measure generalization to an
independent simulation. Hungarian matching minimizes total Euclidean distance;
strength errors use the same assignment. No prediction clipping is applied.
Source width was fixed during inverse training. Final map time: {t[-1]:.6g}.

![True and predicted source locations](source_localization.png)
"""
    (output / "report.md").write_text(text)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    mesh = ax.pcolormesh(x, y, final_field, shading="auto", cmap="inferno")
    ax.scatter([s["x_s"] for s in true_sources], [s["y_s"] for s in true_sources], marker="o", s=180, facecolors="none",
               edgecolors="cyan", linewidths=2, label="True source")
    ax.scatter([s["x_s"] for s in estimates], [s["y_s"] for s in estimates], marker="x", s=90, c="lime", linewidths=2, label="Predicted source")
    for pair in matching["pairs"]:
        a,b = true_sources[pair["true_index"]], estimates[pair["predicted_index"]]
        ax.plot([a["x_s"],b["x_s"]], [a["y_s"],b["y_s"]], color="white", linestyle="--")
    ax.set(xlabel="x (domain units)", ylabel="y (domain units)", aspect="equal",
           title=f"Source localization error = {location:.5f}\nFinal numerical concentration, t = {t[-1]:.3f}")
    ax.legend(loc="lower right")
    fig.colorbar(mesh, ax=ax, label="Reference concentration")
    fig.savefig(output / "source_localization.png", dpi=180)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/inverse_pinn"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "evaluation"
    report = evaluate_run(args.run_dir, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
