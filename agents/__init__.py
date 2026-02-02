"""Pipeline agents for PII detection and anonymization."""


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DetectionCoordinator",
    "create_detection_coordinator",
    "route_after_detection",
]
