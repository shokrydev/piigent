"""Multi-Dimensional Evaluation System.

This module provides comprehensive evaluation beyond P/R/F1:
- Span matching (exact and partial)
- Type matching
- Boundary error analysis
- Confidence calibration (ECE)
- Confusion matrices
- Disagreement-based evaluation
- Distribution matching
"""

from evaluation.metrics import MultiDimensionalMetrics
from evaluation.regression import RegressionTestSuite, RegressionReport, TestCase
from evaluation.disagreement import (
    DisagreementEvaluator,
    DisagreementCase,
    DisagreementReport,
    DisagreementType,
)
from evaluation.distribution import (
    DistributionMatcher,
    Distribution,
    DistributionComparison,
)

__all__ = [
    # Metrics
    "MultiDimensionalMetrics",
    # Regression
    "RegressionTestSuite",
    "RegressionReport",
    "TestCase",
    # Disagreement
    "DisagreementEvaluator",
    "DisagreementCase",
    "DisagreementReport",
    "DisagreementType",
    # Distribution
    "DistributionMatcher",
    "Distribution",
    "DistributionComparison",
]
