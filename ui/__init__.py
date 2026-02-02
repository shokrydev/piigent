"""Human-in-the-Loop UI Components.

Provides Gradio-based interfaces for:
- Validation of flagged cases
- Active learning labeling
- Prompt review and approval
"""

from ui.hitl_interface import (
    create_hitl_interface,
    HITLTrigger,
    AutonomyMode,
)

__all__ = [
    "create_hitl_interface",
    "HITLTrigger",
    "AutonomyMode",
]
