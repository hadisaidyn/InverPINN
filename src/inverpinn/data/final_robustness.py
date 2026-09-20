"""Predefined perturbations and a fail-closed confirmatory advancement gate."""
import json
import numpy as np
import pandas as pd
from inverpinn.data.final_confirmatory import verify,METHODS
from inverpinn.data.synthetic_reference import ROOT,sha256
from inverpinn.evaluation.final_confirmatory import criteria,statistics,validate_rows


def nested_sensors(positions):
    """Farthest-point geometry only; deterministic original-index tie breaks."""
    points=np.asarray(positions,dtype=float)
    if points.shape!=(20,2) or not np.isfinite(points).all():raise ValueError("Require original20 finite sensor coordinates.")
    chosen=[int(np.argmax(((points-points.mean(0))**2).sum(1)))]
    while len(chosen)<20:
        nearest=((points[:,None]-points[chosen])**2).sum(2).min(1);nearest[chosen]=-1
        chosen.append(int(np.argmax(nearest)))
    return {n:chosen[:n] for n in (5,10,20)}


def noisy_observations(clean,fraction,master_seed,index):
    """z+fraction*RMS(z)*Z, same IID standard-normal Z across noise levels.

    No clipping: initial-time noisy readings may contradict the exact known IC.
    This is the declared measurement model, not an altered physical condition.
    """
    if fraction not in (0.,.05,.1,.2):raise ValueError("Undeclared noise level.")
    clean=np.asarray(clean,dtype=float)
    rng=np.random.default_rng(np.random.SeedSequence(master_seed,spawn_key=(index,)))
    noise=rng.standard_normal(clean.shape)
    return clean+fraction*np.sqrt(np.mean(clean**2))*noise


def wind_conditions(u=.35,v=-.15):
    """±10% speed, ±10 degrees CCW rotation; generation wind stays fixed."""
    wind=np.array([u,v]);result={"correct":wind,"speed_minus10":.9*wind,"speed_plus10":1.1*wind}
    for name,degrees in (("direction_minus10",-10.),("direction_plus10",10.)):
        angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
        result[name]=np.array([[c,-s],[s,c]])@wind
    return {k:dict(u=float(w[0]),v=float(w[1])) for k,w in result.items()}


def require_confirmation(root):
    """Recompute gates from sealed raw outcomes; no summary-only bypass."""
    config,_,lock=verify(root)
    decision=json.loads((root/"decision.json").read_text())
    if not decision["passed"]:raise PermissionError("Robustness forbidden: confirmation failed.")
    for name,digest in decision["result_hashes"].items():
        if sha256(root/name)!=digest:raise ValueError("Sealed confirmation results changed.")
    frame=pd.concat([pd.read_csv(root/f"{m}_raw.csv") for m in METHODS],ignore_index=True);validate_rows(frame)
    summaries={m:statistics(frame[frame.method==m],config) for m in METHODS}
    if not all(criteria(summaries,config["advancement"]).values()):raise PermissionError("Confirmation gates do not pass.")
    if sha256(ROOT/"configs/InverPINN_final.yaml")!=lock["J1_hash"]:
        raise ValueError("Final model is not byte-identical to J1.")
    return config
