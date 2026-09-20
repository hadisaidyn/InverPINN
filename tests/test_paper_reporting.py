"""Reporting-only regression checks. Never import/train a neural model or solve a PDE."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_report", ROOT / "scripts/build_paper.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
EVIDENCE = ROOT / "paper/evidence"


@pytest.fixture(scope="module")
def evidence():
    return REPORT.verify_evidence(EVIDENCE)


def test_no_scientific_imports():
    for name in ("build_paper.py", "package_paper_evidence.py"):
        tree = ast.parse((ROOT/"scripts"/name).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports += [n.name for n in node.names]
            if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
        assert not any(n.startswith(("torch", "inverpinn", "scipy.optimize")) for n in imports)


def test_frozen_configs_and_scientific_files_unchanged():
    frozen = REPORT.read_json(EVIDENCE/"final/preregistration.json")
    for name, digest in frozen["code_hashes"].items():
        assert REPORT.sha256(ROOT/name) == digest, name
    for name,key in (("J1_confirmatory.yaml","J1_hash"),("B_revised.yaml","B_revised_hash")):
        assert REPORT.sha256(ROOT/"configs"/name) == frozen[key]


def test_evidence_inventory(evidence):
    manifest,frames=evidence
    assert len(manifest["files"]) == 41
    assert sum(len(f) for f in frames.values()) == 90
    assert manifest["frozen_scientific_commit"] == "eb18380f1488f5c679a0981da75e667fa3ba1b51"


def test_corrupt_evidence_rejected(tmp_path):
    folder=tmp_path/"evidence";shutil.copytree(EVIDENCE,folder)
    (folder/"final/J1_raw.csv").write_text("corrupted")
    with pytest.raises(ValueError,match="Evidence hash mismatch"):
        REPORT.verify_evidence(folder)


def test_all_pairs_observations_and_seeds(evidence):
    _,frames=evidence
    for method,d in frames.items():
        assert d.observations_hash.tolist() == frames["J1"].observations_hash.tolist()
        assert d.optimization_seed.eq(42).all()
        assert d.scenario_id.nunique() == 30
        if method != "classical": assert d.completed_updates.eq(12000).all()


def test_failed_confirmation_and_failure_retention(evidence):
    _,frames=evidence
    result=REPORT.analyze(EVIDENCE,frames)
    assert not result["confirmation_passed"]
    assert result["required_recoveries"] == 27
    assert {m:s["recovered"] for m,s in result["methods"].items()} == {"J1":23,"B_revised":21,"classical":30}
    failures=frames["J1"].loc[~frames["J1"].recovery_success,"scenario_id"]
    assert failures.tolist() == [f"final_blind_v1_{i:03d}" for i in (8,12,16,17,24,27,30)]


def test_distribution_against_frozen_summary(evidence):
    _,frames=evidence
    original=REPORT.read_json(EVIDENCE/"final/summary.json")
    # Frozen final schema includes methods, with continuous distributions.
    for method,d in frames.items():
        for metric in REPORT.METRICS:
            actual=REPORT.distribution(d[metric])
            if actual["count"]:
                expected=original["methods"][method]["metrics"][metric]
                assert actual["median"] == pytest.approx(expected["median"],abs=1e-13)
                assert actual["mean"] == pytest.approx(expected["mean"],abs=1e-13)


def test_headline_numbers(evidence):
    _,frames=evidence
    s=REPORT.analyze(EVIDENCE,frames)["methods"]
    for method,expected in {"J1":(.00557371512496,.0597489421661,.07422785896956),
                            "classical":(.00111123059359,.0121779372727,.0150665475414)}.items():
        for metric,value in zip(REPORT.METRICS[:3],expected):
            assert s[method]["metrics"][metric]["median"] == pytest.approx(value,abs=1e-12)
    assert (s["J1"]["Q_under"],s["J1"]["Q_over"]) == (28,2)
    assert s["J1"]["signed_Q_mean"] == pytest.approx(-.1406382126969,abs=1e-12)
    assert s["J1"]["wilson95"] == pytest.approx([.5907167384,.8820761186],abs=1e-10)


def test_history_not_promoted_or_pooled():
    s=REPORT.read_json(EVIDENCE/"development/summary.json")
    assert s["B_revised_2_created"] is False
    assert s["selected_candidate"] is None
    assert s["candidates"]["J1"]["count"] == 24
    assert s["candidates"]["J1"]["relative_strength_error"]["median"] == pytest.approx(.03372926627152725)
    history=REPORT.read_json(EVIDENCE/"historical/physical_correction_summary.json")
    assert history["B0"]["recovered"]==12
    assert history["B_revised"]["recovered"]==18


def test_conditional_Q_headlines():
    d=pd.read_csv(EVIDENCE/"observability/resolution_bias_results.csv")
    assert d[d.grid==201].relative_Q_error.median()*100 == pytest.approx(.8994028051378423)
    assert d[d.grid==321].relative_Q_error.median()*100 == pytest.approx(.411106481368974)


def test_selected_cases_and_preview(evidence):
    manifest,frames=evidence
    assert REPORT.selected_cases(frames["J1"]) == manifest["selected_cases"] == {
        "success":"final_blind_v1_003","failure":"final_blind_v1_030"}
    for role in ("success","failure"):
        with np.load(EVIDENCE/f"previews/{role}.npz",allow_pickle=False) as p:
            assert p["fields"].shape == p["reference"].shape == (4,161,161)
            np.testing.assert_allclose(p["times"],[.1,.2,.4,.8])


def test_reference_flags_retained(evidence):
    _,frames=evidence
    d=frames["J1"]
    assert d.loc[~d.reference_target_met,"scenario_id"].tolist() == [f"final_blind_v1_{i:03d}" for i in (16,17,27)]
    assert len(d)==30


def test_classical_PDE_missing_not_zero(evidence):
    _,frames=evidence
    assert frames["classical"].pde_rmse.isna().all()
    assert REPORT.distribution(frames["classical"].pde_rmse)["median"] is None
    assert "not evaluated" in (ROOT/"paper/generated/tables/table3_final_benchmark.md").read_text()


def test_preserved_physical_metrics(evidence):
    _,frames=evidence
    for m in ("J1","B_revised"):
        for metric in ("negative_fraction","bc_maxabs","ic_maxabs"):
            assert frames[m][metric].eq(0).all()


def test_existing_destination_rejected(tmp_path):
    with pytest.raises(FileExistsError,match="Refusing to overwrite"):
        REPORT.build(EVIDENCE,tmp_path)


@pytest.mark.parametrize("folder",["results","data","src","configs","paper/evidence"])
def test_protected_destinations_rejected(folder):
    with pytest.raises(ValueError,match="protected tree"):
        REPORT.build(EVIDENCE,ROOT/folder/"reporting_must_not_write_here")


def test_complete_rebuild_and_deterministic_summary(tmp_path,evidence):
    out=tmp_path/"report"
    actual=REPORT.build(EVIDENCE,out)
    assert actual==REPORT.analyze(EVIDENCE,evidence[1])
    assert actual==REPORT.read_json(ROOT/"paper/generated/summary.json")
    for name in (*REPORT.FIGURES,"S1_physics_and_signal"):
        for extension in ("png","pdf"):
            assert (out/"figures"/f"{name}.{extension}").stat().st_size>1000
    assert len(list((out/"tables").glob("table*.md")))==4
    assert len(pd.read_csv(out/"tables/all_final_runs.csv"))==90
    provenance=REPORT.read_json(out/"provenance.json")
    assert not provenance["training_performed"]
    assert not provenance["forward_simulation_performed"]
    for name,digest in REPORT.read_json(out/"artifact_hashes.json").items():
        assert REPORT.sha256(out/name)==digest


def test_canonical_artifact_hashes():
    out=ROOT/"paper/generated"
    for name,digest in REPORT.read_json(out/"artifact_hashes.json").items():
        assert REPORT.sha256(out/name)==digest


@pytest.mark.parametrize("count",[50,100,150])
def test_application_word_counts(count):
    text=(ROOT/"docs/application_description.md").read_text()
    section=text.split(f"## {count}-word description\n")[1].split("\n## ")[0].strip()
    assert len(section.split())==count


def test_final_document_links():
    paths=[ROOT/"README.md",*[(ROOT/"paper"/p) for p in ("README.md","manuscript.md","supplement.md","generated/README.md")],
           *[(ROOT/"docs"/p) for p in ("reproducibility.md","project_summary.md","application_description.md","final_scientific_audit.md")]]
    for path in paths:
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if link.startswith(("https://","http://","#")): continue
            assert (path.parent/link.split("#")[0]).exists(),f"Broken link {path}: {link}"


def test_historical_full_suite_not_misrepresented():
    root=ET.parse(EVIDENCE/"final/final_full_suite.xml").getroot()
    suites=[root] if root.tag=="testsuite" else root.findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites)==451
    assert sum(int(s.attrib["failures"])+int(s.attrib["errors"]) for s in suites)==0
    assert "historical" in (ROOT/"docs/reproducibility.md").read_text()
