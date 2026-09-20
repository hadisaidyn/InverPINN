"""Check public Markdown links against files available in a clean Git checkout.

Standard library only; never fetch external URLs or inspect scientific outcomes.
Sealed evidence is historical provenance, not editable public documentation.
The parser covers this repository's inline/reference links, images, ATX/setext
headings and explicit HTML anchors. It is not a complete Markdown renderer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(r"/Users/|/Desktop/PINN|file://|(?<!\w)[A-Za-z]:[\\/]")


def checkout_files(root: Path) -> set[Path]:
    """Include tracked/new nonignored files, never local ignored artifacts."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return {(root / name).resolve() for name in result.stdout.split("\0") if name}


def without_fences(text: str) -> str:
    """Blank fenced examples while preserving diagnostic line numbers."""
    lines, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence is None and marker:
            fence = marker.group(1)
            lines.append("")
        elif fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    """GitHub-style heading IDs, including duplicates and explicit anchors."""
    text = without_fences(text)
    found = set(re.findall(r'<(?:a|h[1-6])\b[^>]*(?:id|name)=["\']([^"\']+)["\']', text))
    counts: dict[str, int] = {}
    previous = ""
    for line in text.splitlines():
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        heading = match.group(1) if match else None
        if previous.strip() and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line):
            heading = previous.strip()
        if heading is not None:
            heading = re.sub(r"<[^>]+>", "", heading)
            heading = re.sub(r"!?\[([^]]+)\]\([^)]*\)", r"\1", heading)
            slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
            number = counts.get(slug, 0)
            found.add(f"{slug}-{number}" if number else slug)
            counts[slug] = number + 1
        previous = line
    return found


def links(text: str):
    """Yield line number, URL for inline, image and reference-style links."""
    text = without_fences(text)
    definitions = {}
    for line in text.splitlines():
        match = re.match(r"^ {0,3}\[([^]]+)\]:\s*(<[^>]+>|\S+)", line)
        if match:
            definitions[match.group(1).casefold()] = match.group(2).strip("<>")
    for number, line in enumerate(text.splitlines(), 1):
        line = re.sub(r"(`+).*?\1", "", line)
        definition = re.match(r"^ {0,3}\[([^]]+)\]:", line)
        if definition:
            yield number, definitions[definition.group(1).casefold()]
            continue
        for match in re.finditer(r"\]\((<[^>]+>|[^\s)]+)(?:\s+['\"][^\n]*?['\"])?\)", line):
            yield number, match.group(1).strip("<>")
        reference_pattern = r"!?\[([^]]+)\]\[([^]]*)\]"
        for match in re.finditer(reference_pattern, line):
            label = (match.group(2) or match.group(1)).casefold()
            yield number, definitions.get(label, f"MISSING_REFERENCE:{label}")
        # A full reference's second bracket is not another shortcut link.
        line = re.sub(reference_pattern, "", line)
        for match in re.finditer(r"(?<!!)\[([^]]+)\](?![\[(])", line):
            if match.group(1).casefold() in definitions:
                yield number, definitions[match.group(1).casefold()]
        for match in re.finditer(r'<(?:a|img)\b[^>]*(?:href|src)=["\']([^"\']+)["\']', line):
            yield number, match.group(1)


def check(root: Path, files: set[Path]) -> tuple[list[str], int, int]:
    """Reject missing/unpublished targets, bad anchors and local machine paths."""
    errors, count = [], 0
    public = sorted(p for p in files if p.suffix == ".md" and
                    not p.is_relative_to(root / "paper/evidence"))
    directories = {parent for p in files for parent in p.parents if parent.is_relative_to(root)}
    for path in public:
        if not path.exists():
            errors.append(f"{path.relative_to(root)}: missing tracked document")
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if LOCAL_PATH.search(line):
                errors.append(f"{path.relative_to(root)}:{number}: local absolute path")
        for number, url in links(text):
            count += 1
            where = f"{path.relative_to(root)}:{number}"
            if url.startswith("MISSING_REFERENCE:"):
                errors.append(f"{where}: undefined reference {url.split(':', 1)[1]}")
                continue
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.is_relative_to(root) or target not in files | directories or not target.exists():
                errors.append(f"{where}: missing/unpublished target {url}")
            elif parts.fragment and target.suffix == ".md":
                if unquote(parts.fragment) not in anchors(target.read_text(encoding="utf-8")):
                    errors.append(f"{where}: missing anchor {url}")
    return errors, len(public), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors, documents, count = check(ROOT, checkout_files(ROOT))
    for error in errors:
        print(error)
    print(f"Checked {documents} public Markdown files and {count} links; {len(errors)} errors.")
    print("External URLs are syntax/inventory only; sealed historical evidence is excluded.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
