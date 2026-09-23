from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    PASSED = 0
    INVALID = 2
    RUNTIME_FAILURE = 3
    INTERRUPTED = 4
    UNSUPPORTED = 5


class CapabilityAnatomyError(Exception):
    exit_code = ExitCode.RUNTIME_FAILURE
    reason = "runtime_failure"


class InvalidConfigurationError(CapabilityAnatomyError):
    exit_code = ExitCode.INVALID
    reason = "invalid_configuration"


class InvalidEvidenceError(CapabilityAnatomyError):
    exit_code = ExitCode.INVALID
    reason = "invalid_evidence"


class UnsupportedAdapterError(CapabilityAnatomyError):
    exit_code = ExitCode.UNSUPPORTED
    reason = "unsupported_adapter"


class UnsupportedPluginError(CapabilityAnatomyError):
    exit_code = ExitCode.UNSUPPORTED
    reason = "unsupported_plugin"


class InterruptedRunError(CapabilityAnatomyError):
    exit_code = ExitCode.INTERRUPTED
    reason = "interrupted"


class GateAuthorizationError(InvalidEvidenceError):
    def __init__(self, reason: str):
        if reason not in {"gate_authorization_invalid", "gate_approved_source_unavailable", "gate_source_identity_changed"}:
            raise ValueError("unknown gate authorization reason")
        self.reason = reason
        super().__init__(reason)
