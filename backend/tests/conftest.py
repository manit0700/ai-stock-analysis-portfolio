"""Test setup: no network, no real credentials, and a throw-away data folder.

These variables are set before any application module is imported, because the application
reads its configuration once at start-up.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="marketvision-tests-"))
os.environ["MARKETVISION_ENV"] = "test"
os.environ["MARKETVISION_OFFLINE"] = "1"
os.environ["MARKETVISION_DATA_DIR"] = str(_TMP / "data")
os.environ["MARKETVISION_MODEL_DIR"] = str(_TMP / "models")
os.environ["MARKETVISION_ENABLE_FINBERT"] = "0"
for _name in (
    "FINNHUB_API_KEY", "FRED_API_KEY", "NEWSAPI_KEY", "OPENAI_API_KEY", "MARKETVISION_LLM_API_KEY",
    "MARKETVISION_LLM_BASE_URL", "MARKETVISION_LLM_MODEL", "MARKETVISION_NEWSAPI_PAID_PLAN", "MARKETVISION_TOOL_TOKEN", "MARKETVISION_ADMIN_TOKEN", "MARKETVISION_SNAPSHOT_ID",
):
    os.environ.pop(_name, None)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def settings_env(monkeypatch):
    """Change environment variables for one test and have the application notice."""
    from app.core.config import reset_settings

    def apply(**values: str | None) -> None:
        for key, value in values.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        reset_settings()

    yield apply
    monkeypatch.undo()
    reset_settings()


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Each test starts with a clean rate limiter and the real clock."""
    from app.core import clock, security
    from app.marketdata.clients import reset_shared_clients
    from app.services import finnhub_service, newsapi_service

    security.limiter.reset()
    clock.set_clock(None)
    reset_shared_clients()
    finnhub_service.clear_cache()
    newsapi_service.clear_cache()
    yield
    clock.set_clock(None)
    reset_shared_clients()
