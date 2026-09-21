"""Read-only deployment checks; never import scientific models or run solvers.

Secret scanning is a bounded, heuristic release check, not a security guarantee.
Only filenames and finding types are reported, never matching credential values.
Use --history before first publication to include all blobs reachable from HEAD.
Use --smoke for a Streamlit AppTest render and all 30 scenario selections.
"""

from __future__ import annotations

import argparse
import ast
import importlib.abc
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.evidence import IDS, METRICS, load_bundle, load_fields, require

RUNTIME_CODE = ("demo/app.py", "demo/evidence.py", "demo/plots.py", "demo/__init__.py")
FORBIDDEN = {"inverpinn", "torch", "scipy", "sklearn", "xarray", "netCDF4"}
PATTERNS = {
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b",
    "aws-access-key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "google-api-key": r"\bAIza[A-Za-z0-9_-]{35}\b",
    "service-token": r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})\b",
    "credential-url": r"https?://[^\s/:]+:[^\s/@]+@[^\s]+",
    "literal-credential": r"""(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|secret[_-]?access[_-]?key|password)\s*[:=]\s*["']([^"'\r\n]{12,})["']""",
}


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args])


def secret_findings(content):
    """Return types only. Recognizable example placeholders are not credentials."""
    text = content.decode("utf-8", errors="replace")
    findings = []
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            value = match.group(1) if kind == "literal-credential" else match.group(0)
            if kind in {"literal-credential", "credential-url"} and re.search(
                r"(?i)example|placeholder|dummy|your[_ -]|<|\$\{|os\.environ", value
            ):
                continue
            findings.append(kind)
            break
    return sorted(set(findings))


def scan_repository(root=ROOT, history=False):
    """Scan publishable files and optionally reachable history without printing values."""
    names = (
        git("ls-files", "--cached", "--others", "--exclude-standard", "-z", root=root)
        .decode()
        .split("\0")
    )
    findings = []
    for name in sorted(set(filter(None, names))):
        path = root / name
        if path.is_file():
            for kind in secret_findings(path.read_bytes()):
                findings.append({"file": name, "type": kind})
        if Path(name).name in {"secrets.toml", ".env", "id_rsa", "id_ed25519"}:
            findings.append({"file": name, "type": "credential-file"})
    count = 0
    if history:
        objects = git("rev-list", "--objects", "HEAD", root=root).decode().splitlines()
        request = "".join(line.split(" ", 1)[0] + "\n" for line in objects).encode()
        info = (
            subprocess.check_output(
                ["git", "-C", str(root), "cat-file", "--batch-check"], input=request
            )
            .decode()
            .splitlines()
        )
        for original, record in zip(objects, info):
            oid, kind, *_ = record.split()
            if kind != "blob":
                continue
            count += 1
            name = original.partition(" ")[2] or "unnamed-blob"
            for finding in secret_findings(git("cat-file", "blob", oid, root=root)):
                findings.append({"file": f"history:{name}", "type": finding})
    return {
        "files_scanned": len(set(filter(None, names))),
        "history_blobs_scanned": count,
        "findings": findings,
    }


def runtime_audit(root=ROOT):
    """Reject nonportable runtime literals and imports of training/solver interfaces."""
    external = set()
    for name in RUNTIME_CODE:
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                external.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                require(
                    not re.search(
                        r"/Users/|/home/|Desktop/|file://|localhost|127\.0\.0\.1|[A-Za-z]:\\",
                        node.value,
                    ),
                    f"Local runtime path: {name}",
                )
                require(
                    not re.search(
                        r"(?:^|[/'\"])results/|artifacts/|\.pt\b|\.pth\b", node.value
                    ),
                    f"Archive/checkpoint runtime path: {name}",
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(
                    node.func.id not in {"eval", "exec", "__import__"},
                    f"Dynamic execution: {name}",
                )
    require(
        not external & (FORBIDDEN | {"subprocess"}),
        "Scientific/process runtime dependency",
    )
    return sorted(external - sys.stdlib_module_names - {"demo"})


def check(root=ROOT, history=False):
    require(root.resolve() == ROOT, "Run this script from the checkout being checked")
    data, manifest = load_bundle()
    expected = {
        "J1": ("0.005574", "5.975", "7.423"),
        "classical": ("0.001111", "1.218", "1.507"),
    }
    for method, strings in expected.items():
        medians = data["summary"][method]["metrics"]
        actual = (
            f"{medians[METRICS[0]]['median']:.6f}",
            f"{100*medians[METRICS[1]]['median']:.3f}",
            f"{100*medians[METRICS[2]]['median']:.3f}",
        )
        require(actual == strings, "Frozen headline mismatch")
    require(
        data["summary"]["J1"]["Q_under"] == 28 and data["summary"]["J1"]["Q_over"] == 2,
        "Frozen Q bias mismatch",
    )
    require(
        data["required"] == 27 and len(data["scenarios"]) == 30,
        "Frozen confirmation threshold changed",
    )
    runtime_imports = runtime_audit(root)
    require(
        set(runtime_imports) == {"numpy", "pandas", "plotly", "streamlit", "yaml"},
        "Unexpected viewer dependencies",
    )
    pins = {}
    for line in (root / "demo/requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name] = version
    require(
        set(pins) == {"numpy", "pandas", "plotly", "streamlit", "PyYAML"},
        "Non-viewer dependency",
    )
    config = tomllib.loads((root / ".streamlit/config.toml").read_text())
    require(
        not {"address", "port", "baseUrlPath", "enableCORS", "enableXsrfProtection"}
        & config.get("server", {}).keys(),
        "Nonportable/unsafe server setting",
    )
    tracked = set(git("ls-files", "-z", root=root).decode().split("\0"))
    assets = set(RUNTIME_CODE) | {
        "demo/requirements.txt",
        ".streamlit/config.toml",
        "demo/data/manifest.json",
        "paper/evidence/manifest.json",
        "paper/generated/summary.json",
        "paper/generated/artifact_hashes.json",
        "paper/InverPINN_Paper.pdf",
        "docs/reproducibility.md",
    }
    assets.update("demo/data/" + name for name in manifest["files"])
    assets.update(
        "paper/evidence/" + name for name in data["provenance"]["input_hashes"]
    )
    figures = sorted((root / "paper/generated/figures").glob("0[1-8]_*.png"))
    require(len(figures) == 8, "Missing gallery figure")
    assets.update(str(p.relative_to(root)) for p in figures)
    for name in sorted(assets):
        require(
            name in tracked and (root / name).is_file(),
            f"Missing/untracked runtime asset: {name}",
        )
    loaded_bytes = 0
    for sid in IDS:
        loaded_bytes += sum(a.nbytes for a in load_fields(sid).values())
    bundle_bytes = sum(
        p.stat().st_size for p in (root / "demo/data").rglob("*") if p.is_file()
    )
    require(bundle_bytes < 16 * 1024**2, "Demo bundle unexpectedly large")
    require(
        max((root / name).stat().st_size for name in assets) < 50 * 1024**2,
        "Oversized runtime asset",
    )
    security = scan_repository(root, history)
    require(
        not security["findings"],
        "Secrets scan flagged file/type only: " + json.dumps(security["findings"]),
    )
    citation = (root / "CITATION.cff").read_text()
    require(
        "given-names: Khadis" in citation and "family-names: Aidyn" in citation,
        "Unconfirmed author metadata",
    )
    require(
        "Khadis Aidyn" in (root / "README.md").read_text(), "README author mismatch"
    )
    return dict(
        entrypoint="demo/app.py",
        scenario_count=30,
        recovered={"J1": 23, "classical": 30},
        required=27,
        bundle_bytes=bundle_bytes,
        all_decoded_field_array_bytes=loaded_bytes,
        runtime_assets=len(assets),
        imports=runtime_imports,
        requirements=pins,
        secrets_scan=security,
        ignored_file_dependency=False,
        training_dependency=False,
        simulation_dependency=False,
        readiness=True,
    )


def smoke():
    """Exercise all cases in Streamlit's runtime with scientific imports blocked."""

    class RejectScience(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in FORBIDDEN:
                raise RuntimeError("Scientific import attempted by viewer")

    guard = RejectScience()
    sys.meta_path.insert(0, guard)
    try:
        from streamlit.testing.v1 import AppTest

        before = time.perf_counter()
        app = AppTest.from_file(str(ROOT / "demo/app.py")).run(timeout=30)
        startup = time.perf_counter() - before
        require(not app.exception, "App startup exception")
        times = []
        for sid in IDS:
            before = time.perf_counter()
            app.selectbox(key="scenario").set_value(sid).run(timeout=30)
            times.append(time.perf_counter() - before)
            require(
                not app.exception and app.selectbox(key="scenario").value == sid,
                "Scenario smoke failed: " + sid,
            )
            require(
                not any(
                    "verification failed" in e.value.lower()
                    or "field unavailable" in e.value.lower()
                    for e in app.error
                ),
                "Asset/integrity error: " + sid,
            )
        import resource

        # macOS reports bytes; Linux reports KiB. This is total process peak RSS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(
            startup_seconds=startup,
            scenario_switch_median_seconds=sorted(times)[len(times) // 2],
            scenario_switch_max_seconds=max(times),
            scenarios_loaded=len(times),
            process_peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
        )
    finally:
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = check(history=args.history)
        if args.smoke:
            result["smoke"] = smoke()
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Deployment ready: NO\n{error}", file=sys.stderr)
        return 1
    print(
        "Demo entrypoint: OK\nFrozen benchmark metrics: OK\nScenario count: 30\nAssets: OK\nPortable paths: OK"
    )
    print(
        "Secrets scan: OK (heuristic, not a guarantee)\nIgnored-file dependency: NONE\nTraining dependency: NONE\nSimulation dependency: NONE"
    )
    print(json.dumps(result, indent=2))
    print("Deployment ready: YES (code/data; hosting authentication is separate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
