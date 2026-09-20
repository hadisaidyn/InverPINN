"""Publication PNG/PDF summaries; all cases remain visible, no cherry-picking."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS={"J0":"#333333","J1":"#0072B2","J2":"#D55E00","J3":"#009E73"}


def save(fig,root,name,dpi):
    fig.savefig(root/f"{name}.png",dpi=dpi,bbox_inches="tight")
    fig.savefig(root/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def distributions(ax,frame,column,ylabel,log=False):
    names=list(COLORS)
    groups=[frame.loc[frame.candidate==name,column].dropna().to_numpy() for name in names]
    ax.boxplot(groups,tick_labels=names,showfliers=False)
    for i,(name,group) in enumerate(zip(names,groups),1):
        ax.scatter(i+np.linspace(-.12,.12,len(group)),group,s=17,color=COLORS[name],alpha=.75)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y",alpha=.2)


def figures(root,frame,information,config):
    dpi=config["plot_dpi"]
    fig,axes=plt.subplots(1,4,figsize=(13,3.3),layout="constrained")
    q_upper=max(1.6,1.05*frame.Q_pred.max(),1.05*frame.Q_true.max())
    for ax,name in zip(axes,COLORS):
        rows=frame[frame.candidate==name]
        ax.scatter(rows.Q_true,rows.Q_pred,color=COLORS[name],s=25)
        ax.plot([0,q_upper],[0,q_upper],"k--",lw=1)
        ax.set(xlabel="True Q",ylabel="Estimated Q",title=name+(" (hybrid)" if name=="J3" else ""),xlim=(.75,1.45),ylim=(0,q_upper))
    save(fig,root,"Q_true_vs_predicted_candidates",dpi)
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    distributions(ax,frame,"signed_Q_error","Signed Q error (predicted − true)")
    ax.axhline(0,color="k",ls="--",lw=1)
    save(fig,root,"signed_Q_error_distribution",dpi)
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    distributions(ax,frame,"localization_error","Euclidean localization error",True)
    ax.axhline(.08,color="k",ls="--",lw=1,label="Secondary recovery threshold")
    ax.legend(fontsize=8);save(fig,root,"candidate_localization",dpi)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,col,label in zip(axes,["relative_l2","rmse","mae"],["Concentration relative L2","Concentration RMSE","Concentration MAE"]):
        distributions(ax,frame,col,label,True)
    save(fig,root,"candidate_concentration_error",dpi)
    fig,axes=plt.subplots(4,3,figsize=(12,12),layout="constrained")
    for row_axes,identifier in zip(axes,config["trajectory_ids"]):
        for name in ("J0","J1","J2"):
            path=root/"runs"/name/identifier/"seed_42/training_history.jsonl"
            history=pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])
            for ax,key in zip(row_axes,["x_s","y_s","Q"]):
                ax.plot(history.epoch,history[f"source/{key}"],color=COLORS[name],lw=.7,alpha=.8,label=name)
            if name in ("J1","J2"):
                final=frame[(frame.candidate==name)&(frame.scenario_id==identifier)].iloc[0]
                row_axes[2].scatter([12000],[final.Q_pred],s=22,color=COLORS[name],zorder=5)
        truth=frame[(frame.candidate=="J0")&(frame.scenario_id==identifier)].iloc[0]
        for ax,key in zip(row_axes,["x","y","Q"]):
            ax.axhline(truth[f"{key}_true"],color="k",ls="--",lw=.8,label="Truth (evaluation only)")
            ax.set(xlabel="Optimizer update",ylabel=key,title=identifier)
        row_axes[0].legend(fontsize=7)
        row_axes[2].set_title(identifier+"; dots: terminal Q profile",fontsize=9)
    save(fig,root,"source_parameter_trajectories",dpi)
    joined=frame.merge(information[["scenario_id","information_group"]],on="scenario_id",validate="many_to_one")
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for row_axes,group in zip(axes,["lower","higher"]):
        for ax,col,label in zip(row_axes,["localization_error","relative_strength_error","relative_l2"],["Localization","Relative Q error","Relative L2"]):
            distributions(ax,joined[joined.information_group==group],col,label,True)
            ax.set_title(f"{group.capitalize()} sensor-RMS half (n=12)")
    save(fig,root,"weak_vs_strong_information",dpi)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    distributions(axes[0],frame,"pde_rmse","PDE residual RMS",True)
    axes[0].set_title("J3 uses native centered FD, not autograd",fontsize=9)
    distributions(axes[1],frame,"negative_fraction","Fraction C<0")
    for name in COLORS:
        rows=frame[frame.candidate==name]
        axes[2].scatter([name],[rows.bc_maxabs.max()],color=COLORS[name],marker="o")
        axes[2].scatter([name],[rows.ic_maxabs.max()],color=COLORS[name],marker="x")
    axes[2].set(ylabel="Maximum violation",title="BC (circles), IC (crosses)")
    save(fig,root,"physics_diagnostics",dpi)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    for name in COLORS:
        rows=frame[frame.candidate==name]
        axes[0].scatter(rows.runtime_seconds,rows.localization_error,s=20,label=name,color=COLORS[name])
        axes[1].scatter(rows.runtime_seconds,rows.relative_strength_error,s=20,label=name,color=COLORS[name])
    for ax,ylabel in zip(axes,["Localization error","Relative Q error"]):
        ax.set(xlabel="Inference runtime (s; J3 includes J0)",ylabel=ylabel,yscale="log")
        ax.legend()
    save(fig,root,"runtime_tradeoff",dpi)
    # Fixed-index qualitative cases for every method, including hybrid plots.
    positions=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    for identifier in config["trajectory_ids"]:
        for name in COLORS:
            row=frame[(frame.candidate==name)&(frame.scenario_id==identifier)].iloc[0]
            run=root/"runs"/name/identifier/"seed_42"
            if not row.evaluation_success:
                continue
            with np.load(run/"predictions.npz") as stored:
                pred,ref=stored["fields"][-1],stored["reference"][-1]
            fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout="constrained")
            for ax,values,title in zip(axes,[ref,pred,abs(pred-ref)],["Reference","Prediction","Absolute error"]):
                artist=ax.imshow(values,origin="lower",extent=(0,1,0,1),cmap="magma",vmin=0,vmax=max(ref.max(),pred.max()) if title!="Absolute error" else None)
                ax.scatter(positions[:,0],positions[:,1],s=8,color="cyan",label="Sensors")
                ax.scatter(row.x_true,row.y_true,marker="*",s=70,color="lime",label="True source")
                ax.scatter(row.x_pred,row.y_pred,marker="x",s=40,color="white",label="Estimated source")
                ax.set(title=title,xlabel="x",ylabel="y");fig.colorbar(artist,ax=ax,shrink=.75)
            axes[0].legend(fontsize=6);fig.suptitle(f"{name} — {identifier} — t=0.8")
            save(fig,run,"reconstruction",dpi)
