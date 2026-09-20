"""Diagnostic-only publication plots; no fitting or source selection."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def save(fig,root,name,config):
    """Refuse plot replacement; retain both raster and vector output."""
    for extension in ("png","pdf"):
        path=root/f"{name}.{extension}"
        if path.exists():
            raise FileExistsError(f"Existing diagnostic plot preserved: {path}")
    fig.tight_layout()
    for extension in ("png","pdf"):
        fig.savefig(root/f"{name}.{extension}",dpi=config["plot_dpi"],bbox_inches="tight")
    plt.close(fig)


def map_panel(ax,maps,key,title,sensors,log=True):
    """Display the allowed source box while keeping every fixed sensor visible."""
    axis=np.sort(maps.x.unique())
    array=maps.sort_values(["y","x"])[key].to_numpy(dtype=float).reshape(len(axis),len(axis))
    finite=array[np.isfinite(array)&(array>0)]
    norm=LogNorm(vmin=finite.min(),vmax=finite.max()) if log and len(finite) and finite.max()>finite.min() else None
    artist=ax.pcolormesh(axis,axis,array,shading="nearest",norm=norm,cmap="viridis",rasterized=True)
    ax.scatter(sensors[:,0],sensors[:,1],marker="^",s=28,facecolors="white",edgecolors="black",linewidths=.7,label="20 fixed sensors")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Source x (synthetic length)",ylabel="Source y (synthetic length)",title=title,aspect="equal")
    plt.colorbar(artist,ax=ax,shrink=.8,label=key+(" (log color)" if norm else ""))
    return artist


def information_figures(root,config,maps,frames,combined):
    """Written before PINN fitting: forward observability and classical evidence."""
    sensors=np.asarray(json.loads((root/"sensor_geometry.json").read_text()))
    specs=[("sensor_signal_map","sensor_rms","Sensor RMS at candidate Q=1.1"),
           ("Q_sensitivity_map","Q_sensitivity_norm","Scaled Q sensitivity"),
           ("jacobian_condition_map","jacobian_condition","Scaled Jacobian condition; not signal strength"),
           ("observability_map","sigma_min","Minimum scaled singular value; not posterior uncertainty")]
    for name,key,title in specs:
        fig,ax=plt.subplots(figsize=(7,6));map_panel(ax,maps,key,title,sensors);ax.legend(loc="lower left",fontsize=8)
        save(fig,root,name,config)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,key in zip(axes,["x_sensitivity_norm","y_sensitivity_norm"]):
        map_panel(ax,maps,key,key.replace("_"," "),sensors)
    save(fig,root,"location_sensitivity_map",config)
    resolution=frames["resolution_bias_results"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for _,group in resolution.groupby("scenario_id"):
        group=group.sort_values("grid")
        axes[0].plot(group.grid,100*group.signed_relative_Q_error,color="gray",alpha=.3,lw=.8)
        axes[1].plot(group.grid,100*group.relative_Q_error,color="gray",alpha=.3,lw=.8)
    med=resolution.groupby("grid").median(numeric_only=True)
    axes[0].plot(med.index,100*med.signed_relative_Q_error,"o-",color="black",label="Median")
    axes[1].plot(med.index,100*med.relative_Q_error,"o-",color="black",label="Median")
    for ax in axes:
        ax.axhline(0,color="black",ls=":",lw=.7);ax.set_xlabel("Inference grid points per axis");ax.legend()
    axes[0].set_ylabel("Conditional signed Q error (%)");axes[1].set_ylabel("Conditional absolute relative Q error (%)")
    save(fig,root,"Q_error_vs_resolution",config)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,["localization_error","relative_strength_error","jacobian_condition"]):
        ax.scatter(combined.sensor_rms,combined[key],s=25)
        ax.set(xlabel="Observed sensor RMS",ylabel=key,title="Prospective diagnostic_v2; classical / local information")
    save(fig,root,"sensor_signal_vs_classical_error",config)
    envelope=frames["envelope_reference"]
    active=envelope[envelope.scope=="active"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for t,group in active.groupby("time"):
        axes[0].scatter(group.x_true,group.required_positive_latent_q95,label=f"t={t:g}",s=20)
        axes[1].scatter(group.y_true,group.required_positive_latent_q95,s=20)
        axes[2].scatter(group.x_true,group.transform_gain_q05,s=20)
    axes[0].legend();axes[0].set(xlabel="True source x",ylabel="Required C/E, active-region q95")
    axes[1].set(xlabel="True source y",ylabel="Required C/E, active-region q95")
    axes[2].set(xlabel="True source x",ylabel="Required transform gain, active q05")
    save(fig,root,"constraint_envelope_reference",config)


def final_figures(root,config,combined,chain,categories,envelope,network):
    """Compare frozen PINN to earlier diagnostics without selecting a new model."""
    fig,ax=plt.subplots(figsize=(8,5))
    keys=["Q_true","Q_conditional","Q_classical","Q_PINN"]
    for _,row in chain.iterrows():
        ax.plot(range(4),row[keys].to_numpy(dtype=float),"o-",alpha=.45,lw=.8,ms=3)
    ax.set(xticks=range(4),xticklabels=["True","Oracle-location\nconditional","Classical joint","Frozen PINN"],ylabel="Q (synthetic concentration/time)",title="Same diagnostic cases: comparison chain, not proven causation")
    save(fig,root,"Q_bias_chain",config)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,["localization_error","relative_strength_error","relative_l2"]):
        for method,color in (("classical","C0"),("pinn","C1")):
            ax.scatter(combined.sensor_rms,combined[f"{method}_{key}"],label=method,color=color,s=22)
        ax.set(xlabel="Observed sensor RMS",ylabel=key);ax.legend()
    save(fig,root,"sensor_signal_vs_error",config)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,["jacobian_condition","sigma_min","Q_effective_information"]):
        for method,color in (("classical","C0"),("pinn","C1")):
            ax.scatter(combined[key],combined[f"{method}_localization_error"],label=method,color=color,s=22)
        ax.set(xlabel=key,ylabel="Localization error",xscale="log");ax.legend()
    save(fig,root,"conditioning_vs_error",config)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,["localization_error","relative_strength_error","relative_l2"]):
        a,b=combined[f"classical_{key}"],combined[f"pinn_{key}"]
        top=max(a.max(),b.max())
        ax.scatter(a,b,s=25);ax.plot([0,top],[0,top],"k--",lw=.8)
        ax.set(xlabel="Classical "+key,ylabel="PINN "+key)
    save(fig,root,"classical_vs_PINN",config)
    fig,ax=plt.subplots(figsize=(10,4))
    counts=categories.evidence_category.value_counts().sort_index()
    ax.barh(counts.index,counts.values);ax.set(xlabel="Diagnostic scenarios",title="Evidence labels are not established causal mechanisms")
    save(fig,root,"failure_categories",config)
    reference=envelope[(envelope.scope=="active")&(envelope.time==.8)]
    actual=network[(network.scope=="active")&(network.time==.8)]
    both=reference.merge(actual,on="scenario_id",suffixes=("_reference","_network"))
    both=both.merge(combined[["scenario_id","pinn_relative_strength_error"]],on="scenario_id")
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].scatter(both.required_positive_latent_q95,both.positive_latent_q95,c=both.pinn_relative_strength_error,cmap="magma")
    maximum=max(both.required_positive_latent_q95.max(),both.positive_latent_q95.max())
    axes[0,0].plot([0,maximum],[0,maximum],"k--",lw=.8)
    axes[0,0].set(xlabel="Required C/E, active q95",ylabel="PINN softplus(f), active q95")
    axes[0,1].scatter(both.required_raw_latent_q95,both.raw_latent_q95)
    axes[0,1].set(xlabel="Required raw f, active q95",ylabel="PINN raw f, active q95")
    axes[1,0].scatter(both.transform_gain_q05_reference,both.pinn_relative_strength_error,label="Required gain")
    axes[1,0].scatter(both.transform_gain_q05_network,both.pinn_relative_strength_error,label="Actual gain",marker="x")
    axes[1,0].set(xlabel="Active-region transform gain q05",ylabel="PINN relative Q error");axes[1,0].legend()
    axes[1,1].scatter(both.x_true,both.raw_spatial_gradient_q95,label="raw f")
    axes[1,1].scatter(both.x_true,both.C_spatial_gradient_q95,label="C",marker="x")
    axes[1,1].set(xlabel="Source x",ylabel="Spatial input-gradient norm q95");axes[1,1].legend()
    fig.suptitle("Envelope diagnostics at t=.8; coordinate gradients, not parameter updates")
    save(fig,root,"constraint_envelope_diagnostic",config)
    identifier=config["envelope"]["latent_map_scenario"]
    with np.load(root/"diagnostics"/identifier/"envelope_maps.npz",allow_pickle=False) as reference_maps:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,key in zip(axes,["positive_latent","raw_latent","transform_gain"]):
            image=ax.imshow(reference_maps[key][-1],origin="lower",extent=(0,1,0,1),cmap="viridis")
            ax.set(xlabel="x",ylabel="y",title=f"Required {key}");fig.colorbar(image,ax=ax,shrink=.75)
        fig.suptitle(f"Predetermined {identifier}, t=.8; zero-envelope edges excluded")
        save(fig,root,"constraint_envelope_maps",config)
