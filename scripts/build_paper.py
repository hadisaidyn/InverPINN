"""Rebuild the final paper's figures and tables from sealed saved evidence only.

No InverPINN module, torch model, solver, optimizer or checkpoint is imported.
Numbers are recomputed from per-scenario CSV records, not re-fitted. New output
directories are exclusive; the frozen experiment tree is never written to.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B_revised", "J1", "classical")
COLORS = ("#777777", "#0072B2", "#D55E00")
METRICS = ("localization_error", "relative_strength_error", "relative_l2", "rmse", "mae", "pde_rmse", "runtime_seconds")
FIGURES = ("01_schematic", "02_numerical_convergence", "03_source_locations", "04_source_strength",
           "05_error_distributions", "06_observability", "07_success", "08_failure")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_evidence(evidence):
    """Reject missing/changed evidence and mismatched paired observations."""
    manifest = read_json(evidence / "manifest.json")
    for name, item in manifest["files"].items():
        path = evidence / name
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {name}")
    decision = read_json(evidence / "final/decision.json")
    for name, digest in decision["result_hashes"].items():
        if sha256(evidence / "final" / name) != digest:
            raise ValueError(f"Frozen decision hash mismatch: {name}")
    for name, key in (("J1_confirmatory.yaml", "J1_hash"), ("B_revised.yaml", "B_revised_hash")):
        if sha256(evidence / "configs" / name) != decision[key]:
            raise ValueError(f"Frozen configuration mismatch: {name}")
    frames = {m: pd.read_csv(evidence / "final" / f"{m}_raw.csv").sort_values("scenario_id").reset_index(drop=True)
              for m in METHODS}
    ids = [f"final_blind_v1_{i:03d}" for i in range(1, 31)]
    for method, frame in frames.items():
        if frame.scenario_id.tolist() != ids:
            raise ValueError(f"Missing/duplicated benchmark cases: {method}")
        for column in ("observations_hash", "reference_data_hash", "x_true", "y_true", "Q_true"):
            if not frame[column].equals(frames["J1"][column]):
                raise ValueError(f"Paired data mismatch: {method}/{column}")
        if not frame.optimization_seed.eq(42).all():
            raise ValueError("Primary seed changed")
        # The archived final experiment has 90 evaluable runs. Do not drop rows
        # or invent replacement estimates if that evidence changes.
        if not frame.evaluation_success.all() or not frame.computational_success.all():
            raise ValueError("Unexpected failed evaluation; preserve evidence and audit, never drop it")
        loc = np.hypot(frame.x_pred - frame.x_true, frame.y_pred - frame.y_true)
        qerr = abs(frame.Q_pred - frame.Q_true) / abs(frame.Q_true)
        if not (np.allclose(loc, frame.localization_error, atol=1e-13, rtol=1e-12)
                and np.allclose(qerr, frame.relative_strength_error, atol=1e-13, rtol=1e-12)):
            raise ValueError(f"Source metric mismatch: {method}")
        recovery = (loc <= .08) & (qerr <= .20)
        if not np.array_equal(recovery, frame.recovery_success):
            raise ValueError("Recovery threshold/records disagree")
    if decision["passed"] or decision["criteria"]["joint_recovery"]:
        raise ValueError("The final failed confirmation must not be reclassified")
    return manifest, frames


def distribution(values):
    """Finite empirical distribution; missing metrics remain explicitly missing."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, **{k: None for k in ("mean", "median", "sample_sd", "q25", "q75", "q90", "worst")}}
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                sample_sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                q25=float(np.quantile(a,.25)), q75=float(np.quantile(a,.75)),
                q90=float(np.quantile(a,.90)), worst=float(a.max()))


def wilson(k, n):
    """Two-sided 95% Wilson score interval for the scenario recovery fraction."""
    z = NormalDist().inv_cdf(.975)
    p = k/n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def selected_cases(j1):
    successful = j1[j1.recovery_success].sort_values(["localization_error", "scenario_id"])
    return {"success": successful.iloc[len(successful)//2].scenario_id,
            "failure": j1.sort_values(["localization_error", "scenario_id"], ascending=[False,True]).iloc[0].scenario_id}


def analyze(evidence, frames):
    protocol = yaml.safe_load((evidence / "final/protocol.yaml").read_text())
    summary = {"methods": {}, "paired": {}, "selected_cases": selected_cases(frames["J1"]),
               "confirmation_passed": False, "required_recoveries": protocol["advancement"]["min_recovered"]}
    for method, d in frames.items():
        k = int(d.recovery_success.sum())
        signed = d.Q_pred-d.Q_true
        summary["methods"][method] = dict(n=len(d), recovered=k, recovery_rate=k/len(d), wilson95=wilson(k,len(d)),
            metrics={m: distribution(d[m]) for m in METRICS}, Q_under=int((signed < -1e-6).sum()),
            Q_over=int((signed > 1e-6).sum()), Q_equal=int((abs(signed) <= 1e-6).sum()),
            signed_Q_mean=float(signed.mean()), signed_Q_median=float(signed.median()),
            max_negative_fraction=float(d.negative_fraction.max()))
    # Reporting-only paired percentile bootstrap: same resampled scenario indices
    # for all metrics/comparisons. Not a new hypothesis test or selection rule.
    rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"])
    indices = rng.integers(0,30,(protocol["analysis"]["bootstrap_samples"],30))
    for control in ("B_revised", "classical"):
        summary["paired"][control] = {}
        for metric in METRICS:
            a,b = frames["J1"][metric].to_numpy(), frames[control][metric].to_numpy()
            if not np.isfinite(a-b).all():
                continue
            delta = a-b
            tolerance = 1e-8 + 1e-6*np.maximum(abs(a),abs(b))
            summary["paired"][control][metric] = dict(mean=float(delta.mean()),median=float(np.median(delta)),
                mean_bootstrap95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),
                J1_better=int((delta < -tolerance).sum()), control_better=int((delta > tolerance).sum()),
                tied=int((abs(delta) <= tolerance).sum()))
    summary["bootstrap"] = dict(seed=protocol["analysis"]["bootstrap_seed"], resamples=len(indices),
        method="paired scenario percentile bootstrap of the mean difference; common resampled indices")
    return summary


def table(output, name, columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output / "tables" / f"{name}.csv", index=False)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    (output / "tables" / f"{name}.md").write_text("\n".join(lines)+"\n")


def make_tables(output, evidence, frames, summary):
    table(output,"table1_physics",["Quantity","Value","Meaning"],[
        ["Domain / duration","[0,1]² / [0,0.8]","Synthetic length/time units"],
        ["u; v; D","0.35; -0.15; 0.005","L/T; L/T; L²/T"],
        ["Source","one Gaussian; sigma=0.08","Q is peak source intensity (C/T)"],
        ["x_s; y_s; Q ranges","[0.25,0.75]; [0.25,0.75]; [0.8,1.4]","IID uniform final design"],
        ["Reference grid / dt","641² / 0.00003125","dx=dy=0.0015625; 25600 updates"],
        ["Refinement grids / dt","161² / 0.0005; 321² / 0.000125","Richardson estimate, not a bound"],
        ["Inference FD grid / dt","201² / 0.00025","Classical and conditional-Q diagnosis"],
        ["Sensors / observations","20 fixed / 81 times","0 to 0.8; zero added noise"],
        ["Dense evaluation","641² nodes at 0.1,0.2,0.4,0.8","Equal point weights"],
        ["IC / BC","C=0 / C=0 on all four edges","Absorbing Dirichlet; not no-flux"],
        ["Reference accuracy flags","3/30 retained","Cases 016,017,027 exceed 1% target"]])
    table(output,"table2_methods",["Setting","B_revised","J1","Classical"],[
        ["Field","3×64 tanh MLP","same MLP","201² explicit FD"],
        ["Constraints","hard nonnegative, zero IC/BC","same","nonnegative monotone scheme, zero IC/BC"],
        ["Location","sigmoid; Adam","sigmoid; Adam","bounded nonlinear least squares"],
        ["Q","softplus; Adam","analytical PDE profile","analytical observation profile"],
        ["Budget","12000 Adam updates","12000 + terminal Q profile","3 starts; max_nfev=60 each"],
        ["Step size / seed","0.001 / 42","0.001 / 42","deterministic starts from observations"],
        ["Batch sizes","data/PDE=512; IC=128; BC=64/edge","same; terminal profile=8192","all 1620 observations"],
        ["Loss weights","data=10; PDE=IC=BC=1, scaled","same","sensor squared mismatch"],
        ["Inference data","identical sparse observations","identical sparse observations","identical sparse observations"]])
    rows=[]
    for method in METHODS:
        s=summary["methods"][method]; m=s["metrics"]
        rows.append([method,f'{s["recovered"]}/30',f'{m["localization_error"]["median"]:.6f}',
            f'{100*m["relative_strength_error"]["median"]:.3f}%',f'{100*m["relative_l2"]["median"]:.3f}%',
            "not evaluated (different representation)" if m["pde_rmse"]["median"] is None else f'{m["pde_rmse"]["median"]:.6f}'])
    table(output,"table3_final_benchmark",["Method","Joint recovery","Median localization","Median Q error","Median L2","Median PDE RMS"],rows)
    table(output,"table4_limits",["Assumption","What is not established"],[
        ["2D constant known transport","Terrain-resolved flow, 3D mixing, uncertain meteorology"],
        ["One known-width stationary Gaussian","Multiple, moving, arbitrary or unknown-width emitters"],
        ["Zero noise; one fixed sensor layout","Noise robustness or sensor-network generalization"],
        ["Synthetic source and numerical concentration","Verified Almaty emissions or causal attribution"],
        ["No chemical or deposition terms","PM2.5 transformation/removal dynamics"],
        ["Richardson reference estimates","Certified mathematical accuracy bounds"],
        ["30 cases; one primary optimization seed","Universal reliability or initialization robustness"],
        ["Shared discretization family for FD solvers","Independence from transport-model misspecification"]])
    pd.concat(frames.values(), ignore_index=True).to_csv(output/"tables/all_final_runs.csv",index=False)
    pd.DataFrame([dict(method=m,metric=k,**v) for m,s in summary["methods"].items() for k,v in s["metrics"].items()]).to_csv(output/"tables/distributions.csv",index=False)
    pd.DataFrame([dict(control=c,metric=k,**v) for c,s in summary["paired"].items() for k,v in s.items()]).to_csv(output/"tables/paired_effects.csv",index=False)
    for source,name in (("development/candidate_summary.csv","development_candidates.csv"),
                        ("numerical/numerical_convergence.csv","numerical_convergence.csv")):
        (output/"tables"/name).write_bytes((evidence/source).read_bytes())


def save(fig, output, name):
    for ext in ("png","pdf"):
        kwargs = {"metadata":{"CreationDate":None,"ModDate":None}} if ext == "pdf" else {}
        fig.savefig(output/"figures"/f"{name}.{ext}",dpi=300,bbox_inches="tight",**kwargs)
    plt.close(fig)


def make_figures(output,evidence,frames,summary):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"ps.fonttype":42,"figure.constrained_layout.use":True})
    sensors=np.asarray(read_json(evidence/"final/sensor_geometry.json"))
    fig,ax=plt.subplots(figsize=(9,2.8));ax.axis("off")
    labels=["Unknown source\n$(x_s,y_s,Q)$","Known transport\n2D PDE","20 fixed sensors\n81 times; no noise","Inverse inference\nPINN or classical","Estimated source\ncompare with truth"]
    for i,label in enumerate(labels):
        ax.text(.1+.2*i,.63,label,ha="center",va="center",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.6",fc="#EDF3F7",ec="#526777"),fontsize=8)
        if i<4: ax.annotate("",xy=(.205+.2*i,.63),xytext=(.175+.2*i,.63),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    ax.text(.5,.18,"Final blind confirmation: J1 23/30 < required 27/30   |   Classical 30/30\nTruth is available to evaluation, not fitting.",ha="center",transform=ax.transAxes)
    save(fig,output,FIGURES[0])
    conv=pd.read_csv(evidence/"numerical/numerical_convergence.csv")
    fig,axes=plt.subplots(1,3,figsize=(9,2.8))
    studies=[("spatial_advection_diffusion","Upwind + diffusion","dx"),("spatial_diffusion","Diffusion only","dx"),("temporal","Explicit Euler","dt")]
    for ax,(kind,title,x) in zip(axes,studies):
        d=conv[conv.study_type==kind].sort_values(x)
        if d.empty: raise ValueError(f"Missing convergence series {kind}")
        ax.loglog(d[x],d.L2_error,"o-",color=COLORS[1]);ax.set(xlabel=x,ylabel="L2 error",title=title)
        # Explicit compact ticks avoid crowded automatic log-minor labels.
        ax.set_xticks(d[x], labels=[f"{v:.3g}" for v in d[x]], fontsize=7)
        ax.set_yticks(d.L2_error, labels=[f"{v:.2e}" for v in d.L2_error], fontsize=7)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(.05,.9,f'Finest order: {d.iloc[0].empirical_order:.3f}',transform=ax.transAxes,fontsize=8)
        ax.grid(alpha=.25,which="both")
    save(fig,output,FIGURES[1])
    fig,axes=plt.subplots(1,3,figsize=(9,3.2))
    for ax,method,color in zip(axes,METHODS,COLORS):
        d=frames[method]
        ax.scatter(sensors[:,0],sensors[:,1],marker="^",s=14,c="#999999",label="sensor")
        ax.scatter(d.x_true,d.y_true,s=20,facecolors="none",edgecolors="black",label="true")
        ax.scatter(d.x_pred,d.y_pred,s=14,c=color,label="estimated")
        for r in d.itertuples(): ax.plot([r.x_true,r.x_pred],[r.y_true,r.y_pred],c=color,lw=.6)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",title=method,aspect="equal")
    axes[0].legend(fontsize=6,loc="upper left");save(fig,output,FIGURES[2])
    fig,axes=plt.subplots(1,2,figsize=(7,3.3),sharex=True,sharey=True)
    for ax,method,color in zip(axes,("J1","classical"),COLORS[1:]):
        d=frames[method];s=summary["methods"][method]
        ax.plot([0,1.5],[0,1.5],"--",c="black",lw=.8,label="identity")
        ax.scatter(d.Q_true,d.Q_pred,c=color,s=24)
        ax.set(xlabel="True Q",ylabel="Estimated Q",xlim=(.75,1.45),ylim=(0,1.5),
               title=f'{method}: {s["Q_under"]} under / {s["Q_over"]} over')
    save(fig,output,FIGURES[3])
    fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    for ax,metric,label,scale,threshold in zip(axes,METRICS[:3],("Localization error","Q relative error (%)","Concentration L2 (%)"),(1,100,100),(.08,20,None)):
        data=[frames[m][metric].to_numpy()*scale for m in METHODS]
        ax.boxplot(data,tick_labels=["B_rev","J1","Classical"],showfliers=False,widths=.5)
        for i,(values,c) in enumerate(zip(data,COLORS),1):
            ax.scatter(i+np.linspace(-.13,.13,len(values)),values,s=12,c=c,alpha=.8,zorder=3)
        if threshold: ax.axhline(threshold,c="black",ls=":",lw=.8)
        ax.set_ylabel(label);ax.set_ylim(bottom=0)
    save(fig,output,FIGURES[4])
    obs=pd.read_csv(evidence/"observability/map_metrics.csv")
    fig,axes=plt.subplots(1,2,figsize=(8,3.6))
    for ax,col,label in zip(axes,("sensor_rms","sigma_min"),("Sensor RMS (C units)","Smallest scaled singular value")):
        grid=obs.pivot(index="y",columns="x",values=col)
        im=ax.pcolormesh(grid.columns,grid.index,grid.values,shading="nearest",cmap="viridis")
        ax.scatter(sensors[:,0],sensors[:,1],c="white",edgecolors="black",marker="^",s=25)
        ax.quiver(.78,.17,.14,-.06,angles="xy",scale_units="xy",scale=1,width=.004)
        ax.text(.75,.21,"wind",fontsize=8)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",aspect="equal",title=label)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle("Fixed-Q=1.1 diagnostic map; blank area outside candidate-location grid\nLocal sensitivity, not a posterior or global uniqueness proof",fontsize=9)
    save(fig,output,FIGURES[5])
    for role,name in (("success",FIGURES[6]),("failure",FIGURES[7])):
        case=summary["selected_cases"][role];row=frames["J1"].set_index("scenario_id").loc[case]
        with np.load(evidence/f"previews/{role}.npz",allow_pickle=False) as p:
            ref,pred=p["reference"][-1],p["fields"][-1];x,y=p["x"],p["y"]
        fig,axes=plt.subplots(1,3,figsize=(9,3.5))
        vmax=max(float(ref.max()),float(pred.max()))
        for ax,field,title in zip(axes,(ref,pred,abs(ref-pred)),("Reference","J1 prediction","Absolute error")):
            im=ax.pcolormesh(x,y,field,shading="nearest",cmap="magma",vmin=0,vmax=None if title=="Absolute error" else vmax,rasterized=True)
            ax.scatter(sensors[:,0],sensors[:,1],s=15,marker="^",c="white",edgecolors="#777777",linewidths=.4,label="sensor")
            ax.scatter(row.x_true,row.y_true,s=65,marker="*",c="#56B4E9",edgecolors="black",label="true source")
            ax.scatter(row.x_pred,row.y_pred,s=38,marker="x",c="#009E73",linewidths=1.5,label="J1 source")
            ax.set(xlabel="x",ylabel="y",title=title,aspect="equal");fig.colorbar(im,ax=ax,shrink=.75,label="C")
        axes[0].legend(fontsize=5.5,loc="upper left")
        fig.suptitle(f'{case}: {role}; t=0.8\nFull-grid scores: localization={row.localization_error:.4f}, Q error={100*row.relative_strength_error:.1f}%, L2={100*row.relative_l2:.1f}%',fontsize=9)
        save(fig,output,name)
    # Supplement: preserve the physical and weak-signal comparison prominently.
    obs_final=pd.read_csv(evidence/"final/observability.csv")
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    for m,c in zip(METHODS,COLORS):
        d=frames[m].merge(obs_final[["scenario_id","sensor_rms"]],on="scenario_id",validate="one_to_one")
        axes[0].scatter(d.sensor_rms,100*d.relative_strength_error,s=16,c=c,label=m)
    axes[0].axhline(20,c="black",ls=":");axes[0].set(xlabel="Sensor RMS",ylabel="Q relative error (%)",title="Descriptive signal association");axes[0].legend(fontsize=6)
    for i,m in enumerate(METHODS[:2]): axes[1].scatter(np.full(30,i),frames[m].pde_rmse,s=13,c=COLORS[i])
    axes[1].set(xticks=[0,1],xticklabels=METHODS[:2],ylabel="Independent PDE RMS",title="Independent PDE residual")
    axes[2].axis("off");axes[2].text(.05,.65,"B_revised and J1:\nnegative fraction = 0\nBC maximum = 0\nIC maximum = 0\n\nClassical continuous PDE\nresidual: not evaluated.",va="center")
    save(fig,output,"S1_physics_and_signal")


def build(evidence,output):
    start=time.perf_counter()
    evidence,output=Path(evidence).resolve(),Path(output).resolve()
    # A reporting destination must never be within frozen archives/source.
    for protected in (ROOT/"results",ROOT/"data",ROOT/"src",ROOT/"configs",evidence):
        if output.is_relative_to(protected):
            raise ValueError(f"Output would enter protected tree: {protected}")
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    manifest,frames=verify_evidence(evidence)
    summary=analyze(evidence,frames)
    if summary["selected_cases"] != manifest["selected_cases"]:
        raise ValueError("Mechanical case selection mismatch")
    output.mkdir(parents=True,exist_ok=False)
    (output/"figures").mkdir();(output/"tables").mkdir()
    write_json(output/"summary.json",summary)
    make_tables(output,evidence,frames,summary)
    make_figures(output,evidence,frames,summary)
    git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)
    write_json(output/"provenance.json",dict(frozen_scientific_commit=manifest["frozen_scientific_commit"],
        reporting_commit=git,reporting_working_tree=status,reporter_sha256=sha256(__file__),
        evidence_manifest_sha256=sha256(evidence/"manifest.json"),python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ("numpy","pandas","matplotlib","PyYAML")},
        runtime_seconds=time.perf_counter()-start,training_performed=False,forward_simulation_performed=False,
        metric_policy="Full-grid archived scores; original failures retained; no best-of-seed selection",
        figure_policy="300 dpi PNG + vector PDF; archived stride-4 previews only for fields"))
    write_json(output/"artifact_hashes.json",{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",type=Path,default=ROOT/"paper/evidence")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"paper/reproduced")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:
        verify_evidence(args.evidence_dir.resolve());print("PASS: evidence hashes, paired observations, metrics and failed decision verified")
    else:
        result=build(args.evidence_dir,args.output_dir)
        print(f'Paper rebuilt without training. J1: {result["methods"]["J1"]["recovered"]}/30; required 27/30; confirmation FAILED.')


if __name__=="__main__":
    main()
