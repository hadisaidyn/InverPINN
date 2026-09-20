"""All-case confirmation plots, paired comparisons and mechanical worst cases."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inverpinn.visualization.single_source_benchmark import save_figure

COLORS={"J1":"#0072B2","B_revised":"#555555","classical":"#009E73"}


def box(ax,frame,key,label):
    groups=[frame.loc[frame.method==m,key].dropna().to_numpy() for m in COLORS]
    ax.boxplot(groups,tick_labels=list(COLORS),showfliers=False)
    for i,(m,g) in enumerate(zip(COLORS,groups),1):
        ax.scatter(i+np.linspace(-.12,.12,len(g)),g,s=15,alpha=.75,color=COLORS[m])
    ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)


def figures(root,frame,paired,worst,dpi):
    positions=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,m in zip(axes,COLORS):
        rows=frame[frame.method==m]
        ax.scatter(*positions.T,marker="^",s=13,c="grey",label="Sensors")
        ax.scatter(rows.x_true,rows.y_true,facecolors="none",edgecolors="black",s=24,label="True")
        ax.scatter(rows.x_pred,rows.y_pred,marker="x",c=COLORS[m],s=24,label="Estimated")
        for r in rows.itertuples():ax.plot([r.x_true,r.x_pred],[r.y_true,r.y_pred],color=COLORS[m],lw=.7,alpha=.5)
        ax.set(xlim=(0,1),ylim=(0,1),aspect="equal",xlabel="x",ylabel="y",title=m);ax.legend(fontsize=7)
    save_figure(fig,root/"true_vs_predicted",dpi)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    upper=max(1.5,1.05*frame.Q_pred.max())
    for ax,m in zip(axes,COLORS):
        rows=frame[frame.method==m];ax.scatter(rows.Q_true,rows.Q_pred,c=COLORS[m],s=25)
        ax.plot([0,upper],[0,upper],"k--",lw=1);ax.set(xlabel="True Q",ylabel="Estimated Q",title=m,xlim=(.75,1.45),ylim=(0,upper))
    save_figure(fig,root/"Q_true_vs_predicted",dpi)
    for key,name,label in (("signed_Q_error","signed_Q_error","Signed Q error (predicted − true)"),
                           ("localization_error","localization_distribution","Euclidean localization error"),
                           ("relative_strength_error","Q_error_distribution","Absolute relative Q error")):
        fig,ax=plt.subplots(figsize=(6,4),layout="constrained");box(ax,frame,key,label)
        if key=="signed_Q_error":ax.axhline(0,color="black",ls="--")
        else:ax.set_yscale("log")
        save_figure(fig,root/name,dpi)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,key,label in zip(axes,["relative_l2","rmse","mae"],["Concentration relative L2","RMSE","MAE"]):
        box(ax,frame,key,label);ax.set_yscale("log")
    save_figure(fig,root/"concentration_error_distribution",dpi)
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout="constrained")
    a=frame[frame.method=="B_revised"].set_index("scenario_id");b=frame[frame.method=="J1"].set_index("scenario_id").loc[a.index]
    for ax,key,label in zip(axes,["localization_error","relative_strength_error","relative_l2","pde_rmse"],["Localization","Q relative error","L2","PDE RMS"]):
        ax.scatter(a[key],b[key],s=23,c=COLORS["J1"]);v=np.concatenate([a[key].dropna(),b[key].dropna()]);v=v[v>0]
        if len(v):ax.plot([v.min(),v.max()],[v.min(),v.max()],"k--",lw=1)
        ax.set(xscale="log",yscale="log",xlabel="B_revised",ylabel="J1",title=label)
    save_figure(fig,root/"paired_comparison",dpi)
    for row in worst:
        run=root/"runs/J1"/row["scenario_id"]
        if not row["evaluation_success"]:continue
        with np.load(run/"predictions.npz") as data:pred,ref=data["fields"][-1],data["reference"][-1]
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
        for ax,array,label in zip(axes,[ref,pred,abs(pred-ref)],["Reference C","J1 C","Absolute error"]):
            artist=ax.imshow(array,origin="lower",extent=(0,1,0,1),cmap="magma",vmin=0,vmax=max(ref.max(),pred.max()) if label!="Absolute error" else None)
            ax.scatter(*positions.T,c="cyan",s=9,label="Sensors");ax.scatter(row["x_true"],row["y_true"],c="lime",marker="*",s=70,label="True source")
            ax.scatter(row["x_pred"],row["y_pred"],c="white",marker="x",s=45,label="Estimated source")
            ax.set(title=label,xlabel="x",ylabel="y");fig.colorbar(artist,ax=ax,shrink=.8)
        axes[0].legend(fontsize=7);fig.suptitle(f"{row['scenario_id']}, t=0.8; selected by localization rank")
        save_figure(fig,run/"worst_case_reconstruction",dpi)
