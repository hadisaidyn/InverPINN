"""Lightweight publication checks; no model import, fitting or PDE simulation."""
import csv
import importlib.util
import re
from pathlib import Path
from statistics import median
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts/check_docs.py")
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


def test_all_public_markdown_links():
    errors, documents, links = DOCS.check(ROOT, DOCS.checkout_files(ROOT))
    assert documents >= 40
    assert links >= 200
    assert errors == []


def test_heading_ids_and_duplicate_anchors():
    text = "# A heading\n## A heading\n## Q: *bias*?\n## Résumé\n## `Q_profile`\n"
    text += "```md\n# Not a heading\n```\nSetext title\n============\n<a id='custom'></a>"
    assert DOCS.anchors(text) == {"a-heading", "a-heading-1", "q-bias", "résumé",
                                  "q_profile", "setext-title", "custom"}


def test_inline_reference_image_and_html_links():
    text = """[paper](paper.md#intro) ![figure](<a b.png>)
[reference][doc] [doc][] [doc]
[doc]: other.md "Title"
<a href="third.md">Third</a><img src="image.png">
`[example](not-a-link.md)`
~~~md
[example](also-not-a-link.md)
~~~
"""
    targets = [url for _, url in DOCS.links(text)]
    assert targets.count("other.md") == 4
    assert set(targets) == {"paper.md#intro", "a b.png", "other.md", "third.md", "image.png"}


def test_clean_checkout_rejects_ignored_local_artifacts(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("[ignored](results.csv) ![missing](missing.png) [anchor](#absent)")
    # The file exists locally but is not in the clean-checkout inventory.
    (tmp_path / "results.csv").write_text("local archive")
    errors, _, _ = DOCS.check(tmp_path, {readme})
    assert len(errors) == 3
    assert any("missing anchor" in e for e in errors)
    assert any("results.csv" in e for e in errors)


def test_valid_relative_encoded_paths_and_anchors(tmp_path):
    readme = tmp_path / "README.md"
    other = tmp_path / "a b.md"
    readme.write_text("[file](a%20b.md#header) [directory](.) [web](https://example.org/#not-checked)")
    other.write_text("# Header")
    assert DOCS.check(tmp_path, {readme, other})[0] == []


def test_undefined_reference_and_path_escape(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("[missing][definition] [escape](../private.txt)")
    errors, _, _ = DOCS.check(tmp_path, {path})
    assert len(errors) == 2
    assert any("undefined reference" in e for e in errors)


@pytest.mark.parametrize("path", ["/Users/example/project", "file:///private/file", r"C:\local\file"])
def test_absolute_path_detection(path):
    assert DOCS.LOCAL_PATH.search(path)
    assert not DOCS.LOCAL_PATH.search("https://example.org/docs")


def test_historical_evidence_is_not_rewritten_as_public_docs(tmp_path):
    archive = tmp_path / "paper/evidence/final/protocol.md"
    archive.parent.mkdir(parents=True)
    archive.write_text("[historical output](ignored-original.csv)")
    assert DOCS.check(tmp_path, {archive}) == ([], 0, 0)


def test_citation_is_yaml_with_supported_metadata():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    # Public name explicitly confirmed during final delivery, not inferred from Git.
    assert citation["authors"] == [{"given-names": "Khadis", "family-names": "Aidyn"}]
    assert citation["version"] == project["version"]
    assert citation["title"].startswith("InverPINN: Diagnosing Source-Strength Bias")
    assert citation["message"]
    assert not {"doi", "repository-code", "license", "date-released"} & citation.keys()


def test_ci_is_explicitly_report_only():
    # BaseLoader avoids YAML 1.1 interpreting GitHub's `on` key as a boolean.
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["report"]
    assert job["timeout-minutes"] == "10"
    actions = [s for s in job["steps"] if "uses" in s]
    assert all(re.fullmatch(r"actions/(checkout|setup-python)@[0-9a-f]{40}", s["uses"]) for s in actions)
    assert actions[0]["with"]["persist-credentials"] == "false"
    assert actions[1]["with"]["python-version"] == "3.13"
    commands = [line for step in job["steps"] for line in step.get("run", "").splitlines()]
    assert commands == [
        'echo "MPLCONFIGDIR=$RUNNER_TEMP/inverpinn-mpl" >> "$GITHUB_ENV"',
        "python -m pip install -r paper/requirements-report.txt",
        "python -m pip install --no-deps -e .",
        'python -c "import inverpinn; print(\'InverPINN import OK\')"',
        "python scripts/build_paper.py --verify-only",
        "python scripts/check_docs.py",
        "python -m pytest tests/test_paper_reporting.py tests/test_repository_docs.py -q",
        "python scripts/build_paper.py --output-dir paper/reproduced_ci",
    ]


def test_ci_temporary_cache_is_configured_after_runner_allocation():
    """GitHub resolves job env before the runner context is available.

    Set MPLCONFIGDIR through the runner-provided environment file in a step,
    rather than using an invalid job-level runner.temp expression. This
    prevents workflow validation from rejecting all jobs before tests start.
    """
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    for job in workflow["jobs"].values():
        assert all("runner." not in value for value in job.get("env", {}).values())
    report = workflow["jobs"]["report"]
    assert report["env"] == {"MPLBACKEND": "Agg"}
    assert report["steps"][0] == {
        "name": "Configure temporary Matplotlib cache",
        "run": 'echo "MPLCONFIGDIR=$RUNNER_TEMP/inverpinn-mpl" >> "$GITHUB_ENV"',
    }


def test_readme_headlines_match_sealed_raw_results():
    readme = (ROOT / "README.md").read_text()
    for method, label in (("J1", "J1: variable-projection PINN"), ("classical", "Classical inverse")):
        with (ROOT / f"paper/evidence/final/{method}_raw.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 30
        recovered = sum(row["recovery_success"].lower() == "true" for row in rows)
        values = [median(float(r[key]) for r in rows) for key in
                  ("localization_error", "relative_strength_error", "relative_l2")]
        table_row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |"))
        assert f"{recovered}/30" in table_row
        assert f"{values[0]:.6f}" in table_row
        assert f"{values[1]*100:.3f}%" in table_row
        assert f"{values[2]*100:.3f}%" in table_row
        if method == "J1":
            assert f"{recovered/30*100:.2f}%" in table_row
            assert sum(float(r["Q_pred"]) < float(r["Q_true"]) for r in rows) == 28
            assert sum(float(r["Q_pred"]) > float(r["Q_true"]) for r in rows) == 2


def test_failed_confirmation_is_visible_before_first_figure():
    readme = (ROOT / "README.md").read_text()
    first_figure = readme.index("![")
    assert readme.index("J1 failed the preregistered reliability threshold") < first_figure
    assert "27/30 recoveries (90%) were required" in readme[:first_figure]
    assert "## License\n\nNo license has been selected or included." in readme


def test_gallery_reuses_eight_canonical_figures():
    gallery = (ROOT / "docs/figures.md").read_text()
    images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", gallery)
    assert len(images) == 8
    assert all(url.startswith("../paper/generated/figures/") for url in images)
    assert len(set(images)) == 8
