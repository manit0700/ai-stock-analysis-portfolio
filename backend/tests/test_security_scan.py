"""The secret scanner: catches planted keys, passes the clean tree, and the bundle stays clean."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
SCANNER = BACKEND / "scripts" / "scan_secrets.py"


def _scan_text(text: str) -> list[str]:
    sys.path.insert(0, str(BACKEND / "scripts"))
    try:
        import scan_secrets
    finally:
        sys.path.pop(0)
    path = REPO / "tests_tmp_scan.ts"
    try:
        path.write_text(text, encoding="utf-8")
        return scan_secrets.scan_file(path)
    finally:
        path.unlink(missing_ok=True)


def test_the_scanner_catches_the_shapes_that_were_actually_found():
    # the committed Finnhub key's shape (audit S1); value altered — not-a-secret
    assert _scan_text('const KEY = "d1kvaaaa01qtaaaaaaaad1kvaaaa01qtaaaaaag0";'.replace("KEY", "K"))
    assert _scan_text('url = "wss://ws.finnhub.io?token=abcdefghij1234567890"')
    assert _scan_text('OPENAI_API_KEY = "sk-' + "A" * 40 + '"')
    assert _scan_text("-----BEGIN RSA PRIVATE KEY-----")


def test_the_scanner_ignores_placeholders_and_marked_fixtures():
    assert not _scan_text('api_key = "your_api_key_goes_here_example_000"')
    assert not _scan_text('token = "aaaabbbbccccddddeeeeffff0000"  // not-a-secret')
    assert not _scan_text("const price = 123.45;")


def test_the_tracked_tree_has_no_secrets():
    result = subprocess.run([sys.executable, str(SCANNER)], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert result.returncode == 0, result.stderr


def test_a_built_frontend_bundle_has_no_key_shaped_strings():
    static = REPO / "frontend-next" / ".next" / "static"
    if not static.exists():
        import pytest

        pytest.skip("frontend not built on this machine")
    sys.path.insert(0, str(BACKEND / "scripts"))
    try:
        import scan_secrets
    finally:
        sys.path.pop(0)
    findings = []
    for path in static.rglob("*.js"):
        for label, pattern in scan_secrets.PATTERNS:
            if label in {"Finnhub-style key", "OpenAI key", "token in a URL", "private key block"} and pattern.search(path.read_text(errors="ignore")):
                findings.append(f"{path.name}: {label}")
    assert findings == []
