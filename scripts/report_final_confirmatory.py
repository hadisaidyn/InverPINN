"""Apply the frozen final decision after all three methods and independent audit."""
import json
import shutil
import numpy as np
import pandas as pd
from inverpinn.data.final_confirmatory import verify,manifest,METHODS,old_inventory
from inverpinn.data.synthetic_reference import ROOT,sha256,write_json
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.evaluation.final_confirmatory import statistics,criteria,validate_rows,paired,METRICS
from inverpinn.visualization.final_confirmatory import figures
from run_model_diagnosis import snapshot


def records(frame):
    return frame.astype(object).where(pd.notna(frame),None).to_dict("records")


def report(root):
    config,plan,lock=verify(root);manifest(root)
    if (root/"summary.json").exists() or (root/"decision.json").exists():raise FileExistsError("Final decision already exists; no overwrite.")
    generated=("paired_J1_vs_Brevised.csv","paired_J1_vs_classical.csv","worst_J1_cases.csv","information_strata.csv","summary.csv","reporting_execution")
    if any((root/name).exists() for name in generated):raise FileExistsError("Partial report preserved; review before any restart.")
    audit=json.loads((root/"integrity_audit.json").read_text())
    if audit["audited_runs"]!=90 or not audit["execution_code_unchanged"]:raise ValueError("Audit incomplete.")
    for name,digest in audit["result_hashes"].items():
        if sha256(root/name)!=digest:raise ValueError("Audited results changed.")
    if old_inventory()!=json.loads((root/"consumed_hashes.json").read_text()):raise ValueError("Historical artifacts changed.")
    frame=pd.concat([pd.read_csv(root/f"{m}_raw.csv") for m in METHODS],ignore_index=True);validate_rows(frame)
    summaries={m:statistics(frame[frame.method==m],config) for m in METHODS}
    checks=criteria(summaries,config["advancement"]);passed=all(checks.values())
    contrasts={}
    for method,name in (("B_revised","paired_J1_vs_Brevised.csv"),("classical","paired_J1_vs_classical.csv")):
        table,summary=paired(frame,method,config["analysis"]);write_csv(root/name,records(table));contrasts[method]=summary
    info=pd.read_csv(root/"observability.csv");j1=frame[frame.method=="J1"].merge(info,on="scenario_id",validate="one_to_one")
    ordered=j1.sort_values(["evaluation_success","localization_error","scenario_id"],ascending=[True,False,True],na_position="first")
    worst=records(ordered.head(3));write_csv(root/"worst_J1_cases.csv",worst)
    weak=[]
    for (method,group),rows in frame.merge(info[["scenario_id","information_group"]],on="scenario_id").groupby(["method","information_group"]):
        s=statistics(rows,config)
        weak.append(dict(method=method,information_group=group,count=s["count"],recovered=s["recovered"],localization_median=s["metrics"]["localization_error"]["median"],
            Q_error_median=s["metrics"]["relative_strength_error"]["median"],L2_median=s["metrics"]["relative_l2"]["median"]))
    write_csv(root/"information_strata.csv",weak)
    rows=[]
    for method,s in summaries.items():
        for metric,values in s["metrics"].items():rows.append(dict(method=method,metric=metric,**values))
    write_csv(root/"summary.csv",rows)
    bank=json.loads((root/"maps/bank_lock.json").read_text())
    artifacts=dict(protocol_hash=lock["protocol_hash"],J1_hash=lock["J1_hash"],B_revised_hash=lock["B_revised_hash"],
        fitting_archive_hash=sha256(root/"fitting_execution/source_snapshot.zip"),result_hashes=audit["result_hashes"])
    result=dict(confirmation_passed=passed,criteria=checks,methods=summaries,paired=contrasts,worst_J1_cases=worst,information_strata=weak,
        classical_bank_cost=bank,reference_target_misses=int((~frame[frame.method=="J1"].reference_target_met).sum()),
        maximum_reference_error_estimate=float(frame.reference_error_estimate.max()),historical_files_unchanged=audit["historical_files_unchanged"],
        prior_promotion_rule_passed=False,new_fixed_hypothesis=True,InverPINN_final_created=passed,robustness_eligible=passed,**artifacts)
    execution=root/"reporting_execution";execution.mkdir(exist_ok=False);snapshot(execution)
    figures(root,frame,contrasts,worst,config["plot_dpi"])
    final=ROOT/"configs/InverPINN_final.yaml"
    if passed:
        if final.exists():raise FileExistsError("Existing final config must not be overwritten.")
        shutil.copyfile(root/"J1_frozen.yaml",final)
        if sha256(final)!=lock["J1_hash"]:raise ValueError("Final model differs from J1.")
    elif final.exists():raise ValueError("A failed confirmation cannot leave a final model configuration.")
    write_json(root/"summary.json",result)
    write_json(root/"decision.json",dict(passed=passed,criteria=checks,summary_hash=sha256(root/"summary.json"),**artifacts))
    print(json.dumps(dict(passed=passed,criteria=checks,recovered=summaries["J1"]["recovered"]),indent=2),flush=True)


if __name__=="__main__":report(ROOT/"results/final_confirmatory")
