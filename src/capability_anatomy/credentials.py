"""Execution-only credentials. References, never values, belong in evidence."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import hashlib
from threading import Lock
import re
from typing import Any, Mapping
from urllib.parse import urlsplit, parse_qsl

from opentelemetry import context as otel_context, trace as otel_trace
from opentelemetry.trace import Status, StatusCode

from .errors import InvalidConfigurationError
from .telemetry import OperationTelemetry


@dataclass(frozen=True)
class CredentialReference:
    provider: str
    name: str

    def __post_init__(self) -> None:
        if self.provider != "env" or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", self.name) is None:
            raise InvalidConfigurationError("credential reference requires provider env and a valid environment variable name")


class SecretValue:
    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def reveal(self) -> str:
        """Explicit plaintext access for the trusted plugin's transport boundary."""
        return self.__value

    def __repr__(self) -> str:
        return "SecretValue([REDACTED])"

    def __str__(self) -> str:
        return "[REDACTED]"

    def __reduce__(self):
        raise TypeError("runtime credentials cannot be serialized")


class _ResolvedSecrets:
    def __init__(self):
        self._values: set[str] = set()
        self._lock = Lock()

    def add(self, value: str):
        with self._lock:
            self._values.add(value)

    def contains(self, value: str) -> bool:
        with self._lock:
            return any(secret in value for secret in self._values)


_RESOLVED_SECRETS: ContextVar[_ResolvedSecrets | None] = ContextVar("capability_anatomy_resolved_secrets", default=None)


@contextmanager
def credential_scope():
    token = _RESOLVED_SECRETS.set(_ResolvedSecrets())
    try:
        yield
    finally:
        _RESOLVED_SECRETS.reset(token)


def refuse_secret_persistence(value: str) -> None:
    secrets = _RESOLVED_SECRETS.get()
    if secrets is not None and secrets.contains(value):
        signals = OperationTelemetry.create("credentials")
        with signals.tracer.start_as_current_span("capability_anatomy.credentials.persist", record_exception=False, set_status_on_exception=False) as span:
            signals.record(span, operation="persist_evidence", outcome="refused", reason="resolved_credential_persistence_refused")
            span.set_status(Status(StatusCode.ERROR, "resolved_credential_persistence_refused"))
        raise InvalidConfigurationError("resolved credential reached an evidence field; remove credential material from plugin outputs")


class RuntimeCredentials:
    """Resolve references only when the execution consumer calls get()."""
    def __init__(self, references: Mapping[str, CredentialReference], *, telemetry: OperationTelemetry | None = None) -> None:
        self._references = dict(references)
        self._registry = _RESOLVED_SECRETS.get()
        self._context = otel_context.get_current()
        self._telemetry = telemetry or OperationTelemetry.create("credentials")

    def get(self, alias: str) -> SecretValue:
        context = None if otel_trace.get_current_span().get_span_context().is_valid else self._context
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.credentials.resolve", context=context, record_exception=False, set_status_on_exception=False,
        ) as span:
            if isinstance(alias, str):
                span.set_attribute("capability_anatomy.credential_alias_sha256", hashlib.sha256(alias.encode()).hexdigest())
            reference = self._references.get(alias)
            value = os.environ.get(reference.name) if reference is not None else None
            reason = "credential_reference_missing" if reference is None else "credential_environment_missing"
            if not value:
                self._telemetry.record(span, operation="resolve_credential", outcome="refused", reason=reason)
                span.set_status(Status(StatusCode.ERROR, reason))
                raise InvalidConfigurationError(f"{reason}: configure credentials with an environment reference and set its environment variable")
            self._telemetry.record(span, operation="resolve_credential", outcome="accepted", reason="credential_environment_resolved")
            for registered in (self._registry, _RESOLVED_SECRETS.get()):
                if registered is not None:
                    registered.add(value)
            return SecretValue(value)


# Plugin parameter names ending in these authentication-material families are
# reserved. References/settings (e.g. authorization_file, token_limit) are not
# material fields. OAuth's bare "code" is reserved only as a complete name;
# matching its suffix would wrongly classify trust_remote_code.
_SECRET_FAMILIES = frozenset({
    "password", "passwd", "passphrase", "secret", "token", "key", "apikey",
    "privatekey", "authorization", "auth", "cookie", "credential", "credentials",
    "assertion", "signature",
})
_SECRET_COMPOUNDS = frozenset({"access_key_id", "connection_string"})


def _secret_key(key: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower().replace("-", "_")
    parts = normalized.split("_")
    return (normalized == "code" or parts[-1] in _SECRET_FAMILIES
            or any("_".join(parts[index:]) in _SECRET_COMPOUNDS for index in range(len(parts))))


class InlineCredentialError(InvalidConfigurationError):
    pass


def reject_inline_credentials(value: Mapping[str, Any]) -> None:
    """Refuse credential fields and credential-bearing URLs without echoing input."""
    pending: list[Any] = [item for key, item in value.items() if key != "credentials"]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if _secret_key(str(key)):
                    raise InlineCredentialError("inline credential field refused; move credentials to top-level credentials: {alias: {provider: env, name: ENV_NAME}} and use a credential-aware plugin")
                pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            if "://" in item or item.startswith("//"):
                try:
                    url = urlsplit(item)
                    parameter_fields = (url.query, url.fragment.lstrip("?"), url.fragment.partition("?")[2])
                    unsafe = url.username is not None or url.password is not None or any(
                        _secret_key(key) for field in parameter_fields for key, _ in parse_qsl(field)
                    )
                except ValueError:
                    unsafe = True
                if unsafe:
                    raise InlineCredentialError("credential-bearing URL refused; use a credential-free URL and an environment credential reference")
            if re.match(r"(?i)^\s*(bearer|basic)\s+\S+", item) or re.match(r"^(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|IQoJb3JpZ2luX2Vj[A-Za-z0-9+/=]{48,})$", item):
                raise InlineCredentialError("inline credential value refused; use an environment credential reference")
