"""Deployment-only regression checks; no scientific computation or credentials."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_demo_deployment as check


def test_readiness_and_memory_budget():
    report = check.check()
    assert report["readiness"] is True
    assert report["scenario_count"] == 30
    assert report["recovered"] == {"J1": 23, "classical": 30}
    assert report["required"] == 27
    assert report["bundle_bytes"] == 8759214
    assert report["all_decoded_field_array_bytes"] == 9487680
    assert report["runtime_assets"] == 58
    assert report["secrets_scan"]["findings"] == []


def test_all_scenarios_import_and_render_without_science():
    result = check.smoke()
    assert result["scenarios_loaded"] == 30
    assert result["startup_seconds"] > 0
    assert result["process_peak_rss_bytes"] > 0


@pytest.mark.parametrize(
    "bad_code",
    [
        "import torch",
        "from inverpinn.training import train",
        "import scipy.optimize",
        "import subprocess",
        'path="/Users/example/data.npz"',
        'path="results/hidden.npz"',
        'path="artifacts/local.json"',
        'url="http://localhost:8501"',
    ],
)
def test_runtime_audit_rejects_local_or_scientific_dependencies(tmp_path, bad_code):
    for name in check.RUNTIME_CODE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (tmp_path / "demo/app.py").write_text(bad_code)
    with pytest.raises(ValueError):
        check.runtime_audit(tmp_path)


def test_secrets_findings_do_not_include_values():
    fake = "ghp_" + "A" * 36
    findings = check.secret_findings(fake.encode())
    assert findings == ["github-token"] and fake not in str(findings)
    literal = "password" + ' = "' + "not-a-real-password-for-test" + '"'
    assert check.secret_findings(literal.encode()) == ["literal-credential"]
    placeholder = "api_key" + ' = "YOUR_API_KEY_HERE"'
    assert check.secret_findings(placeholder.encode()) == []


def test_missing_asset_fails_loudly(monkeypatch):
    actual = Path.is_file

    def missing(path):
        return False if path == ROOT / "paper/InverPINN_Paper.pdf" else actual(path)

    monkeypatch.setattr(Path, "is_file", missing)
    with pytest.raises(ValueError, match="Missing/untracked runtime asset"):
        check.check()


def test_untracked_runtime_asset_fails_loudly(monkeypatch):
    original = check.git

    def missing(*args, **kwargs):
        result = original(*args, **kwargs)
        if args == ("ls-files", "-z"):
            result = result.replace(b"paper/InverPINN_Paper.pdf\0", b"")
        return result

    monkeypatch.setattr(check, "git", missing)
    with pytest.raises(ValueError, match="Missing/untracked runtime asset"):
        check.check()


def test_header_and_nonexpert_explanation():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "demo/app.py")).run(timeout=30)
    assert not app.exception
    assert any(h.value == "What am I looking at?" for h in app.subheader)
    assert any(
        "Interactive viewer for frozen synthetic" in m.value for m in app.markdown
    )
    assert any("does not identify verified real-world" in m.value for m in app.info)
    assert any("FAILED" in m.value for m in app.error)


def test_secrets_ignored_and_python_documented():
    assert ".streamlit/secrets.toml" in (ROOT / ".gitignore").read_text()
    doc = (ROOT / "demo/DEPLOYMENT.md").read_text()
    assert "Python 3.13" in doc and "Advanced settings" in doc
    assert "demo/app.py" in doc and "main" in doc
