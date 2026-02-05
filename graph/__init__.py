"""LangGraph pipeline definitions for PII detection and anonymization."""

from .state import FlowState, DetectedEntity


def __getattr__(name):
    """Lazy import for flow components with external dependencies."""
    if name == "create_privacy_flow":
        from .privacy_flow import create_privacy_flow
        return create_privacy_flow
    if name == "run_flow":
        from .privacy_flow import run_flow
        return run_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FlowState",
    "DetectedEntity",
    "create_privacy_flow",
    "run_flow",
]
