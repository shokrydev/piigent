"""LangGraph pipeline definitions for PII detection and anonymization."""

from .state import PipelineState, DetectedEntity


def __getattr__(name):
    """Lazy import for pipeline with external dependencies."""
    if name == "create_privacy_pipeline":
        from .pipeline_graph import create_privacy_pipeline
        return create_privacy_pipeline
    if name == "run_pipeline":
        from .pipeline_graph import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineState",
    "DetectedEntity",
    "create_privacy_pipeline",
    "run_pipeline",
]
