"""System-Level Safeguards.

Provides protection against:
- Overfitting to synthetic data artifacts
- Prompt/template leakage
- Regression from changes
"""

from safeguards.overfitting import OverfittingDetector, OverfittingReport
from safeguards.leakage import LeakageChecker, LeakageReport

__all__ = [
    "OverfittingDetector",
    "OverfittingReport",
    "LeakageChecker",
    "LeakageReport",
]
