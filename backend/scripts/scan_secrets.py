"""Scan the repository for credentials that should never be committed.

    python scripts/scan_secrets.py            scan the working tree (exit 1 on a finding)
    python scripts/scan_secrets.py --staged   scan files staged in git (for a pre-commit hook)

Looks for known provider key shapes, generic assignments of long secrets, private key blocks,
and URLs that carry a token. Test fixtures can mark a deliberate fake with ``not-a-secret`` on
the same line. This catches mistakes, not attackers; review is still review.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 40 lowercase alphanumerics with at least one letter beyond f, so git commit hashes (pure hex) do not match.
    ("Finnhub-style key", re.compile(r"\b(?=[a-z0-9]{40}\b)[a-f0-9]*[g-z][a-z0-9]*\b")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("token in a URL", re.compile(r"[?&](?:token|apikey|api_key|apiKey)=[A-Za-z0-9_\-]{12,}")),
    ("assigned secret", re.compile(r"""(?ix)\b(api[_-]?key|apikey|secret|token|password)\s*[:=]\s*["'][A-Za-z0-9+/_\-]{20,}["']""")),
]

SKIP_DIRS = {".git", "node_modules", ".next", "__pycache__", ".venv", "venv", ".pytest_cache", "quarantine"}
SKIP_SUFFIXES = {".png", ".jpg", ".ico", ".woff", ".woff2", ".zip", ".gz", ".joblib", ".sqlite3", ".parquet", ".pdf", ".lock"}
SKIP_FILES = {"package-lock.json", "tsconfig.tsbuildinfo"}
ALLOW_MARKER = "not-a-secret"
#: Words that make an "assigned secret" match a false alarm (schema fields, placeholders).
BENIGN = re.compile(r"(?i)example|placeholder|your[_-]|<\w+>|redacted|dummy|changeme|xxxxx")


def tracked_or_staged(staged: bool) -> list[Path]:
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged else ["git", "ls-files"]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line.strip()]


def scan_file(path: Path) -> list[str]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for number, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        for label, pattern in PATTERNS:
            match = pattern.search(line)
            if match and not (label == "assigned secret" and BENIGN.search(line)):
                shown = match.group(0)
                findings.append(f"{path.relative_to(ROOT)}:{number}: {label}: {shown[:6]}...{shown[-4:] if len(shown) > 10 else ''}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    findings: list[str] = []
    for path in tracked_or_staged(args.staged):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES or path.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        findings.extend(scan_file(path))
    if findings:
        print("Possible secrets found. Remove them (and rotate anything real):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print("No secrets found in the scanned files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
