"""All configuration comes from the environment. Nothing secret lives in source code.

Read once through ``get_settings()``. Tests call ``reset_settings()`` after changing the environment.
See ``backend/.env.example`` for every variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.secrets import Secret

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

_DEFAULT_CORS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    env: str
    data_dir: Path
    model_dir: Path
    # Provider credentials. All optional: the application runs in a reduced mode without them.
    finnhub_api_key: Secret
    fred_api_key: Secret
    newsapi_key: Secret
    llm_api_key: Secret
    llm_base_url: str
    llm_model: str
    # Access control.
    tool_token: Secret
    admin_token: Secret
    cors_origins: tuple[str, ...]
    # Behaviour.
    offline: bool
    enable_finbert: bool
    paper_trading_enabled: bool
    http_timeout_seconds: float
    http_max_retries: int

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "marketvision.sqlite3"

    def capabilities(self) -> dict[str, bool]:
        """What this process can do with the configuration it has. Shown by ``/health``."""
        return {
            "network_providers": not self.offline,
            "finnhub": bool(self.finnhub_api_key) and not self.offline,
            "fred_api": bool(self.fred_api_key) and not self.offline,
            "newsapi": bool(self.newsapi_key) and not self.offline,
            "llm_explanations": bool(self.llm_api_key) or bool(self.llm_base_url),
            "finbert": self.enable_finbert,
            "paper_trading": self.paper_trading_enabled,
            "tool_token_set": bool(self.tool_token),
            "admin_token_set": bool(self.admin_token),
        }


def _load_dotenv() -> None:
    """Load ``backend/.env`` if python-dotenv is installed. Real environment variables win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(BACKEND_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    env = (os.getenv("MARKETVISION_ENV", "development") or "development").strip().lower()
    origins_raw = os.getenv("MARKETVISION_CORS_ORIGINS", "").strip()
    origins = tuple(o.strip().rstrip("/") for o in origins_raw.split(",") if o.strip()) or _DEFAULT_CORS
    return Settings(
        env=env,
        data_dir=_path("MARKETVISION_DATA_DIR", BACKEND_ROOT / "data"),
        model_dir=_path("MARKETVISION_MODEL_DIR", BACKEND_ROOT / "models"),
        finnhub_api_key=Secret(os.getenv("FINNHUB_API_KEY")),
        fred_api_key=Secret(os.getenv("FRED_API_KEY")),
        newsapi_key=Secret(os.getenv("NEWSAPI_KEY")),
        # Any OpenAI-compatible server works, including a local one. No provider is hard-coded.
        llm_api_key=Secret(os.getenv("MARKETVISION_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
        llm_base_url=(os.getenv("MARKETVISION_LLM_BASE_URL", "") or "").strip(),
        llm_model=(os.getenv("MARKETVISION_LLM_MODEL", "") or "gpt-4o-mini").strip(),
        tool_token=Secret(os.getenv("MARKETVISION_TOOL_TOKEN")),
        admin_token=Secret(os.getenv("MARKETVISION_ADMIN_TOKEN")),
        cors_origins=origins,
        offline=_flag("MARKETVISION_OFFLINE", False),
        enable_finbert=_flag("MARKETVISION_ENABLE_FINBERT", False),
        paper_trading_enabled=_flag("MARKETVISION_PAPER_TRADING", True),
        http_timeout_seconds=_number("MARKETVISION_HTTP_TIMEOUT", 10.0),
        http_max_retries=int(_number("MARKETVISION_HTTP_RETRIES", 2)),
    )


def reset_settings() -> None:
    """Forget the cached settings. For tests that change the environment."""
    get_settings.cache_clear()
