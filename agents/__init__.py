"""Pipeline agents for PII detection and anonymization.

Includes:
- Detection coordination and aggregation
- Self-critique and explanation
- Error taxonomy classification
- Fix proposal generation
- Verification
"""


def __getattr__(name):
    """Lazy import for agents with external dependencies."""
    if name == "DetectionCoordinator":
        from .detection_coordinator import DetectionCoordinator
        return DetectionCoordinator
    if name == "create_detection_coordinator":
        from .detection_coordinator import create_detection_coordinator
        return create_detection_coordinator
    if name == "route_after_detection":
        from .detection_coordinator import route_after_detection
        return route_after_detection

    # Self-critique agents
    if name == "RationaleAgent":
        from .rationale import RationaleAgent
        return RationaleAgent
    if name == "ErrorTaxonomyAgent":
        from .error_taxonomy import ErrorTaxonomyAgent
        return ErrorTaxonomyAgent
    if name == "FixProposalAgent":
        from .fix_proposal import FixProposalAgent
        return FixProposalAgent
    if name == "VerificationAgent":
        from .verification import VerificationAgent
        return VerificationAgent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Detection
    "DetectionCoordinator",
    "create_detection_coordinator",
    "route_after_detection",
    # Self-critique
    "RationaleAgent",
    "ErrorTaxonomyAgent",
    "FixProposalAgent",
    "VerificationAgent",
]
