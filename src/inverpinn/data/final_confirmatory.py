"""Revision-8 preregistration, consumed-data exclusion and immutable inputs.

This module never selects a model. IID uniform source draws are frozen once;
truth enters only reference generation and explicitly post-fit evaluation.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from inverpinn.data.development_v3 import previous_plans, development_plan
from inverpinn.data.single_source_benchmark import object_hash, load_frozen, write_csv
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json, provenance

REVISION7 = "33892895805d692c8c3960d8646c43d6a4f7536d"
IDS = tuple(f"final_blind_v1_{i:03d}" for i in range(1,31))
METHODS = ("J1", "B_revised", "classical")


def read_config(name):
    return yaml.safe_load((ROOT/"configs"/name).read_text())


def exact_J1():
    """Extract the executed Revision-7 recipe, not its failed selection rule."""
    b=read_config("B_revised.yaml");r=read_config("inverse_formulation_revision.yaml")
    return dict(name="J1_confirmatory",status="new_fixed_hypothesis_not_previously_promoted",
        origin_commit=REVISION7,
        recipe=dict(training=b["inference"]["training"],formulation=b["inference"]["formulation"],sigma=.08,seed=42),
        candidate=r["candidates"]["J1"],profiling=r["profiling"],
        architecture=b["architecture"],optimizer=b["optimizer"],
        source_Q="analytic_profile_buffer_not_independent_softplus_or_Adam_parameter",
        source_xy="sigmoid",gradient_interval=r["gradient_interval"],
        collocation=b["collocation"],observation_sampling=b["observation_sampling"],
        dtype=b["dtype"],device=b["device"],torch_threads=1,deterministic_algorithms=True)


def old_inventory():
    """Protect all consumed results; exclude only this revision's two roots."""
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"results").rglob("*"))
        if p.is_file() and not any(p.is_relative_to(ROOT/"results"/n)
            for n in ("final_confirmatory","final_robustness"))}


def previous_sources():
    """Plans and truth/config exclusions only; never previous performance."""
    rows,seeds=previous_plans()
    dev=read_config("inverse_formulation_revision.yaml")
    rows+=development_plan(dev);seeds.add(dev["scenario_sampling"]["seed"])
    triples=set()
    def visit(value,key=""):
        if isinstance(value,dict):
            if {"x_s","y_s","Q"}<=value.keys():
                triples.add(object_hash({k:float(value[k]) for k in ("x_s","y_s","Q")}))
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif "seed" in key and type(value) is int:seeds.add(value)
    for p in (ROOT/"configs").glob("*.yaml"):
        if p.name not in ("final_confirmatory.yaml","J1_confirmatory.yaml","InverPINN_final.yaml"):
            visit(yaml.safe_load(p.read_text()))
    for p in (ROOT/"results").rglob("source.json"):
        if "truth" in p.parts and not ({"final_confirmatory","final_robustness"}&set(p.parts)):
            visit(json.loads(p.read_text()))
    return rows,seeds,triples


def scenario_plan(config, previous, used_seeds, used_triples):
    """Independent uniform x,y,Q, one immutable draw per ID; never redraw."""
    spec=config["scenario_sampling"];n=spec["count"];seed=spec["seed"]
    if n!=30 or seed in used_seeds:
        raise ValueError("Final benchmark requires 30 cases and a new seed.")
    rng=np.random.default_rng(seed)
    values=rng.uniform([.25,.25,.8],[.75,.75,1.4],size=(n,3))
    rows=[]
    for identifier,theta,child in zip(IDS,values,np.random.SeedSequence(seed).spawn(n)):
        source=dict(zip(("x_s","y_s","Q"),map(float,theta)))
        if object_hash(source) in used_triples:raise ValueError("Consumed source overlap; no replacement.")
        source["sigma"]=.08
        rows.append(dict(scenario_id=identifier,split="final_blind_v1",source=source,source_hash=object_hash(source),
            generation_seed=int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-1)))
    for key in ("scenario_id","source_hash","generation_seed"):
        new={r[key] for r in rows}
        if len(new)!=n or new & {r[key] for r in previous}:raise ValueError("Consumed or duplicate scenario.")
    return rows


def freeze(config,root):
    """Freeze protocol/config/code before drawing any new source or reference."""
    if root.exists():raise FileExistsError("Final confirmation already exists; preserve it.")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if head!=REVISION7:raise ValueError("Revision-7 HEAD changed.")
    if read_config("J1_confirmatory.yaml")!=exact_J1():raise ValueError("J1 recipe changed.")
    old=read_config("inverse_formulation_revision.yaml");classical=read_config("observability_diagnosis.yaml")
    for key in ("domain","physics","source_family","noise_std","sensors_file","sensor_count","measurement_times","evaluation_times","reference","recovery"):
        if config[key]!=old[key]:raise ValueError(f"Controlled problem changed: {key}")
    for key in ("classical","scaling","sensitivity","observability_map"):
        if config[key]!=classical[key]:raise ValueError(f"Classical diagnostic changed: {key}")
    codes={}
    tracked=subprocess.check_output(["git","ls-tree","-r","--name-only",head,"src","scripts","configs"],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        expected=hashlib.sha256(subprocess.check_output(["git","show",f"{head}:{name}"],cwd=ROOT)).hexdigest()
        if sha256(ROOT/name)!=expected:raise ValueError(f"Frozen prior code/config changed: {name}")
        codes[name]=expected
    for name in (str(Path(__file__).relative_to(ROOT)),"src/inverpinn/evaluation/final_confirmatory.py","src/inverpinn/data/final_robustness.py"):
        codes[name]=sha256(ROOT/name)
    prior,seeds,triples=previous_sources();historical=old_inventory()
    root.mkdir(parents=True)
    (root/"protocol.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copyfile(ROOT/"docs/J1_confirmatory_protocol.md",root/"protocol.md")
    for src,dst in ((config["sensors_file"],"sensor_geometry.json"),("configs/J1_confirmatory.yaml","J1_frozen.yaml"),("configs/B_revised.yaml","B_revised_frozen.yaml")):
        shutil.copyfile(ROOT/src,root/dst)
    pre=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol_hash=object_hash(config),
        protocol_document_hash=sha256(root/"protocol.md"),sensor_layout_hash=sha256(root/"sensor_geometry.json"),
        reference_solver_commit=head,reference_solver_hash=codes["src/inverpinn/physics/finite_difference.py"],
        inference_config_hash=sha256(ROOT/config["reference"]["inference_config"]),code_hashes=codes,
        J1_hash=sha256(root/"J1_frozen.yaml"),B_revised_hash=sha256(root/"B_revised_frozen.yaml"))
    write_json(root/"preregistration.json",pre)
    rows=scenario_plan(config,prior,seeds,triples)
    write_json(root/"scenario_plan.json",rows)
    write_json(root/"protocol_lock.json",pre|dict(plan_hash=object_hash(rows)))
    write_json(root/"consumed_hashes.json",historical)
    write_json(root/"independence_check.json",dict(prior_planned_cases=len(prior),new_cases=30,
        master_seed=config["scenario_sampling"]["seed"],overlap=0,prior_outcomes_used=False,historical_files=len(historical)))
    provenance(root)


def verify(root):
    config,plan,lock=load_frozen(root)
    if sha256(root/"J1_frozen.yaml")!=lock["J1_hash"] or sha256(root/"B_revised_frozen.yaml")!=lock["B_revised_hash"]:
        raise ValueError("Frozen method config changed.")
    if read_config("J1_confirmatory.yaml")!=exact_J1() or sha256(ROOT/"configs/J1_confirmatory.yaml")!=lock["J1_hash"]:
        raise ValueError("J1 no longer equals Revision 7.")
    return config,plan,lock


def manifest(root):
    config,plan,lock=verify(root);rows=[];inputs=[]
    for item in plan:
        case=root/"datasets"/item["scenario_id"];row=json.loads((case/"generation.json").read_text())
        if object_hash(json.loads((case/"truth/source.json").read_text()))!=item["source_hash"] or row["source_hash"]!=item["source_hash"]:
            raise ValueError("Truth disagrees with frozen plan.")
        for key,path in (("observations_hash",case/"inputs/observations.npz"),("physics_hash",case/"inputs/physics.json"),("reference_data_hash",root/row["reference_path"])):
            if sha256(path)!=row[key]:raise ValueError("Immutable input/reference changed.")
        if row["max_reference_error_estimate"] is None or not np.isfinite(row["max_reference_error_estimate"]):
            raise ValueError("Undefined reference quality; preserve and review.")
        row.update(x_true=item["source"]["x_s"],y_true=item["source"]["y_s"],Q_true=item["source"]["Q"],reference_grid=641,dt=.00003125)
        rows.append(row);inputs.append({k:row[k] for k in ("scenario_id","observations_hash","physics_hash")})
    hashes=dict(manifest_hash=object_hash(rows),inputs_hash=object_hash(inputs))
    path=root/"manifest_lock.json"
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=v for k,v in hashes.items()) or sha256(root/"manifest.csv")!=old["csv_hash"] or object_hash(json.loads((root/"input_manifest.json").read_text()))!=old["inputs_hash"]:
            raise ValueError("Frozen manifest changed.")
    else:
        if (root/"manifest.csv").exists() or (root/"input_manifest.json").exists():raise FileExistsError("Partial manifest preserved.")
        write_csv(root/"manifest.csv",rows);write_json(root/"input_manifest.json",inputs)
        write_json(path,hashes|dict(csv_hash=sha256(root/"manifest.csv"),frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    return rows


def check_open(root,path,fitting_case=None,method=None):
    """Accidental truth/consumed-output read guard, not a security sandbox."""
    if isinstance(path,int):return
    root=Path(root).resolve();path=Path(os.fsdecode(path)).resolve()
    if path.is_relative_to(ROOT/"results") and not path.is_relative_to(root):raise PermissionError("Consumed results forbidden.")
    if fitting_case is not None and path.is_relative_to(root):
        allowed=path.is_relative_to(root/"datasets"/fitting_case/"inputs") or path.is_relative_to(root/"runs"/method/fitting_case)
        allowed|=path==root/"maps/unit_responses.npz" and method=="classical"
        if path.name in {"metrics.json","predictions.npz"}:allowed=False
        if not allowed:raise PermissionError("Truth/other results forbidden during fitting.")


def install_guard(root):
    state=dict(fitting_case=None,method=None)
    def audit(event,args):
        if event=="open":check_open(root,args[0],**state)
    sys.addaudithook(audit);return state
