"""Domain errors. Service code raises these; the web layer turns them into HTTP responses.

Keeping ``fastapi`` out of the services means the same code can be called from tests, scripts,
a backtest or an agent tool without a web server.
"""
from __future__ import annotations

from typing import Any


class MarketVisionError(Exception):
    """Base class. ``code`` is stable and machine-readable; ``message`` is for people."""

    code = "error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if code:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationFailed(MarketVisionError):
    code = "validation_failed"
    http_status = 422


class NotFound(MarketVisionError):
    code = "not_found"
    http_status = 404


class DataUnavailable(MarketVisionError):
    """No provider could supply the data. Never answered with an invented value."""

    code = "data_unavailable"
    http_status = 503


class ProviderError(MarketVisionError):
    """One provider failed. The unified service may still succeed through another."""

    code = "provider_error"
    http_status = 502


class RateLimited(ProviderError):
    code = "rate_limited"
    http_status = 429


class NotSupported(MarketVisionError):
    code = "not_supported"
    http_status = 501


class FeatureUnavailable(MarketVisionError):
    """A feature a model needs could not be computed. The model must not be given a substitute."""

    code = "feature_unavailable"
    http_status = 503


class SchemaMismatch(MarketVisionError):
    """Training and serving feature schemas differ. The model must not be used."""

    code = "feature_schema_mismatch"
    http_status = 500


class ModelUnavailable(MarketVisionError):
    code = "model_unavailable"
    http_status = 503


class PermissionDenied(MarketVisionError):
    code = "permission_denied"
    http_status = 403


class AuthRequired(MarketVisionError):
    code = "auth_required"
    http_status = 401


class NotConfigured(MarketVisionError):
    code = "not_configured"
    http_status = 503


class OrderRejected(MarketVisionError):
    """A paper order failed validation. ``code`` says which rule refused it."""

    code = "order_rejected"
    http_status = 409


class Conflict(MarketVisionError):
    code = "conflict"
    http_status = 409
