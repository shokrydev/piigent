"""Benchmarking tools for PIIgent pipeline.

Note: Document generation now uses SynPII. Import directly:
    from synpii import SynPII
    synpii = SynPII(preset="clinical_de")
    doc = synpii.generate_document()
"""

from .metrics import calculate_metrics, calculate_dataset_metrics, MatchResult

__all__ = [
    "calculate_metrics",
    "calculate_dataset_metrics",
    "MatchResult",
]
