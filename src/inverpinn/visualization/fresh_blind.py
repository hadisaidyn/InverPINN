"""All-case fresh benchmark figures; no visual selection or clipped predictions."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from inverpinn.visualization.single_source_benchmark import save_figure


def figures(root, baseline, revised, paired, seeds, config):
    dpi = config["plot_dpi"]
    colors = {"B0":"#0072B2", "B_revised":"#D55E00"}
    sensors = np.array(json.loads((root/"sensor_geometry.json").read_text()))
    fig, ax = plt.subplots(figsize=(6,6), layout="constrained")
    ax.scatter(*sensors.T, marker="^", s=22, c="gray", label="Fixed sensors")
    ax.scatter(revised.x_true, revised.y_true, facecolors="none", edgecolors="black", label="True sources")
    for label, frame in (("B0",baseline), ("B_revised",revised)):
        ax.scatter(frame.x_pred, frame.y_pred, marker="x", c=colors[label], label=label)
        for row in frame.itertuples():
            ax.plot([row.x_true,row.x_pred], [row.y_true,row.y_pred], lw=.5, alpha=.5, c=colors[label])
    ax.set(xlim=(0,1), ylim=(0,1), aspect="equal", xlabel="x (synthetic length)", ylabel="y (synthetic length)", title="All 30 fresh primary scenarios")
    ax.legend(fontsize=8)
    save_figure(fig, root/"fresh_true_vs_predicted_sources", dpi)
    for metric, filename, xlabel in (("localization_error","fresh_localization_distribution","Localization error (length)"),
                                    ("relative_l2","fresh_concentration_error_distribution","Concentration relative L2")):
        fig, axes = plt.subplots(1,2, figsize=(10,4), layout="constrained")
        for label, frame in (("B0",baseline),("B_revised",revised)):
            values = np.sort(frame[metric].dropna())
            axes[0].hist(values, bins=10, histtype="step", color=colors[label], label=label)
            axes[1].step(values, np.arange(1,len(values)+1)/30, where="post", c=colors[label], label=label)
        axes[0].set(xlabel=xlabel,ylabel="Scenario count")
        axes[1].set(xlabel=xlabel,ylabel="Fraction of all 30 scenarios",ylim=(0,1.02))
        axes[0].legend(); axes[1].legend()
        save_figure(fig,root/filename,dpi)
    fig, ax = plt.subplots(figsize=(6,5),layout="constrained")
    maximum = 1.05*max(revised.Q_true.max(), revised.Q_pred.max(), baseline.Q_pred.max())
    ax.plot([0,maximum],[0,maximum],ls="--",c="gray",label="Identity")
    for label, frame in (("B0",baseline),("B_revised",revised)):
        ax.scatter(frame.Q_true,frame.Q_pred,c=colors[label],label=label)
    ax.set(xlabel="True Q (concentration/time)",ylabel="Predicted Q (concentration/time)",title="Fresh primary fits: no calibration")
    ax.legend(); save_figure(fig,root/"fresh_Q_true_vs_predicted",dpi)
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax, key in zip(axes.flat,("localization_error","relative_strength_error","relative_l2","rmse","mae","pde_rmse")):
        ax.axhline(0,color="gray",lw=1)
        ax.scatter(np.arange(1,31),paired[f"delta_{key}"],s=22,c=colors["B_revised"])
        ax.set(xlabel="Fresh scenario index",ylabel=f"Delta {key}",title="Revised − B0; negative is better")
    save_figure(fig,root/"fresh_B0_vs_Brevised_paired",dpi)
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax,key in zip(axes.flat,("pde_rmse","pde_mae","bc_rmse","ic_rmse","negative_fraction","minimum_concentration")):
        for label,frame in (("B0",baseline),("B_revised",revised)):
            ax.plot(np.arange(1,31),frame[key],"o-",ms=3,lw=.6,label=label,c=colors[label])
        ax.set(xlabel="Fresh scenario index",ylabel=key)
    axes.flat[0].legend(fontsize=8)
    save_figure(fig,root/"fresh_physical_diagnostics",dpi)
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    ids = config["optimization"]["additional_seed_scenarios"]
    for j,seed in enumerate([42,*config["optimization"]["additional_seeds"]]):
        frame = seeds[seeds.optimization_seed == seed].set_index("scenario_id").loc[ids]
        for ax,key in zip(axes,("localization_error","Q_pred","relative_l2")):
            ax.scatter(np.arange(10)+(j-1)*.16,frame[key],s=22,label=f"seed {seed}")
            ax.set(xticks=np.arange(10),xticklabels=[s[-3:] for s in ids],xlabel="Preselected fresh ID",ylabel=key)
    axes[0].legend(fontsize=8)
    save_figure(fig,root/"fresh_seed_sensitivity",dpi)
    landscape = pd.read_csv(root/"identifiability/landscapes.csv")
    grid = landscape[landscape.kind == "grid"]
    info = json.loads((root/"identifiability/summary.json").read_text())
    axis = np.sort(grid.x.unique())
    for key, label in (("fixed_Q","Oracle: Q fixed to truth"),("profiled_Q","Q profiled from observations only")):
        values = grid[f"{key}_mse"].to_numpy().reshape(len(axis),len(axis))
        positive = values[values>0]
        norm = LogNorm(vmin=positive.min(),vmax=positive.max()) if positive.size and positive.max()>positive.min() else None
        fig, ax = plt.subplots(figsize=(7,6),layout="constrained")
        mesh=ax.pcolormesh(axis,axis,values,shading="auto",norm=norm,cmap="viridis")
        truth,pred=info["true_source"],info["primary_PINN_source"]
        ax.scatter(truth["x_s"],truth["y_s"],s=90,facecolors="none",edgecolors="red",label="True source")
        if pred:
            ax.scatter(pred["x_s"],pred["y_s"],s=80,marker="x",c="white",label="Primary PINN")
        ax.set(xlabel="Candidate xs",ylabel="Candidate ys",aspect="equal",title=f"fresh_015 — {label}\nSensor MSE, not a posterior probability")
        fig.colorbar(mesh,ax=ax,label="Sensor MSE (concentration²; log color)")
        ax.legend(fontsize=8); save_figure(fig,root/f"fresh_identifiability_{key}",dpi)


def reconstruction(root, row, output, config):
    """Mechanical rank selection is external; this function cannot choose cases."""
    output.mkdir(exist_ok=False)
    # Record even a selected computational failure, never replace its example.
    from inverpinn.data.synthetic_reference import write_json
    write_json(output/"selection.json", row)
    if not row["evaluation_success"]:
        (output/"failure.txt").write_text(row["failure"] or "Evaluation missing; no replacement example.")
        return
    case=root/"datasets"/row["scenario_id"]
    generation=json.loads((case/"generation.json").read_text())
    run=root/"runs/B_revised"/row["scenario_id"]/f"seed_{config['optimization']['primary_seed']}"
    with np.load(root/generation["reference_path"],allow_pickle=False) as data:
        truth=data["fields"][-1]
    with np.load(run/"predictions.npz",allow_pickle=False) as data:
        prediction=data["fields"][-1]
    sensors=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    lo,hi=min(truth.min(),prediction.min()),max(truth.max(),prediction.max())
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for ax,field,title in zip(axes,(truth,prediction,np.abs(truth-prediction)),("Reference C","B_revised C","Absolute error")):
        image=ax.imshow(field,origin="lower",extent=[0,1,0,1],cmap="inferno",**({} if title=="Absolute error" else dict(vmin=lo,vmax=hi)))
        ax.scatter(*sensors.T,c="cyan",s=9,label="Sensors")
        ax.scatter(row["x_true"],row["y_true"],s=70,facecolors="none",edgecolors="lime",label="True source")
        ax.scatter(row["x_pred"],row["y_pred"],s=70,marker="x",c="white",label="Predicted source")
        ax.set(xlabel="x",ylabel="y",title=title)
        fig.colorbar(image,ax=ax,label="Synthetic concentration")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{row['scenario_id']}; primary seed 42; localization={row['localization_error']:.5f}; t=.8")
    save_figure(fig,output/"reconstruction",config["plot_dpi"])
