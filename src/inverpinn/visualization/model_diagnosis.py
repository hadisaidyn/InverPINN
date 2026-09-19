"""Development-only diagnostic plots; never load benchmark metadata or truth."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_figures(run, history, gradients, preview, dpi):
    """Archive loss/source trajectories and un-clipped final field preview."""
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in ("data", "pde", "initial", "boundary"):
        axes[0].plot(frame.epoch, frame[f"weighted/{name}"].clip(lower=1e-30), label=name)
    axes[0].set(yscale="log", xlabel="Adam update", ylabel="Weighted objective contribution")
    axes[0].legend()
    for name in ("x_s", "y_s", "Q"):
        axes[1].plot(frame.epoch, frame[f"source/{name}"], label=name)
    axes[1].set(xlabel="Adam update", ylabel="Physical source parameter")
    axes[1].legend()
    save(fig, run/"training", dpi)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    truth, prediction = preview["truth"][-1], preview["fields"][-1]
    low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
    for ax, title, values in zip(axes, ("Reference", "Prediction (unclipped)", "Absolute error"),
                                  (truth, prediction, abs(truth-prediction))):
        artist = ax.imshow(values, extent=[0, 1, 0, 1], origin="lower", cmap="viridis",
                           vmin=low if title != "Absolute error" else 0,
                           vmax=high if title != "Absolute error" else None)
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(artist, ax=ax)
    save(fig, run/"reconstruction", dpi)


def audit_figures(frame, gradients, output, dpi):
    """Median across the ten cases at each update; no held-out observations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped = frame.groupby("epoch").median(numeric_only=True)
    for name in ("data", "pde", "initial", "boundary"):
        ax.plot(grouped.index, grouped[f"weighted/{name}"], label=name)
    ax.set(xlabel="Adam update", ylabel="Median weighted loss", yscale="log")
    ax.legend()
    save(fig, output/"loss_component_evolution", dpi)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, values in gradients.groupby("component"):
        grouped = values.groupby("epoch").median(numeric_only=True)
        axes[0].plot(grouped.index, grouped.weighted_nn_norm.clip(lower=1e-30), label=name)
        axes[1].plot(grouped.index, grouped.Q_weighted_gradient, label=name)
    axes[0].set(xlabel="Update", ylabel="Weighted NN gradient norm", yscale="log")
    axes[1].set(xlabel="Update", ylabel="Signed weighted dL/dQ (positive pushes down)")
    for ax in axes:
        ax.legend()
    save(fig, output/"gradient_norm_evolution", dpi)


def residual_figure(run, records, maps, times, dpi):
    fig, axes = plt.subplots(1, len(times), figsize=(5*len(times), 4), squeeze=False)
    for ax, values, time in zip(axes[0], maps, times):
        bound = abs(values).max()
        artist = ax.imshow(values, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.scatter([records["x_s"]], [records["y_s"]], marker="x", c="black", label="Estimated source")
        ax.set(title=f"PDE residual at t={time:g}", xlabel="x", ylabel="y")
        ax.legend()
        fig.colorbar(artist, ax=ax)
    save(fig, run/"residual_maps", dpi)


def comparison_figures(rows, summary, output, dpi):
    """Paired development metrics and physical diagnostics; no clipped errors."""
    names = list(summary.candidate)
    labels = dict(localization_error="Localization error", relative_strength_error="Relative Q error",
                  relative_l2="Concentration relative L2")
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,(key,label) in zip(axes,labels.items()):
        for i,name in enumerate(names):
            values=rows[rows.candidate==name][key].to_numpy()
            ax.scatter(np.full(len(values),i),values,s=12,alpha=.65)
        ax.plot(range(len(names)),summary[f"{key}_median"],"k_",markersize=12,label="Median")
        ax.set(xticks=range(len(names)),xticklabels=names,ylabel=label)
        ax.tick_params(axis="x",rotation=65)
    axes[0].legend()
    save(fig,output/"development_baseline_vs_candidates",dpi)
    fig, axes=plt.subplots(1,3,figsize=(15,4))
    baseline=rows[rows.candidate=="B0"].set_index("scenario_id")
    for ax,(key,label) in zip(axes,labels.items()):
        for name in names:
            paired=rows[rows.candidate==name].set_index("scenario_id")[key]-baseline[key]
            ax.plot([names.index(name)]*len(paired),paired,".",alpha=.7)
        ax.axhline(0,c="black",lw=.7)
        ax.set(xticks=range(len(names)),xticklabels=names,ylabel=f"{label}: candidate minus B0")
        ax.tick_params(axis="x",rotation=65)
    save(fig,output/"candidate_metric_summary",dpi)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,key in zip(axes.ravel(),("negative_fraction","pde_rmse","bc_rmse","ic_rmse")):
        ax.plot(names,summary[f"{key}_mean"],"o-",label="Mean")
        ax.plot(names,summary[f"{key}_max"],"x--",label="Maximum across cases")
        ax.set(ylabel=key)
        ax.tick_params(axis="x",rotation=65)
        ax.legend()
    save(fig,output/"physical_violations_by_candidate",dpi)
    columns=5
    fig,axes=plt.subplots(int(np.ceil(len(names)/columns)),columns,figsize=(16,3.2*int(np.ceil(len(names)/columns))))
    for ax,name in zip(axes.ravel(),names):
        subset=rows[rows.candidate==name]
        ax.scatter(subset.Q_true,subset.Q_pred,s=20)
        ax.plot([.8,1.4],[.8,1.4],"k--",lw=.8)
        ax.set(title=name,xlabel="True Q (evaluation only)",ylabel="Predicted Q",xlim=(.75,1.45),
               ylim=(0,max(1.5,float(rows.Q_pred.max())*1.05)))
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    save(fig,output/"Q_true_vs_predicted_development",dpi)
