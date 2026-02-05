"""Overfitting Detection.

Detects when prompts overfit to synthetic data artifacts
rather than learning generalizable patterns.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from evaluation.metrics import MultiDimensionalMetrics


@dataclass
class OverfittingReport:
    """Report on overfitting detection."""
    is_overfitting: bool
    synthetic_real_gap: Dict[str, float]
    gap_threshold: float
    recommendation: str  # "continue", "diversify_synthetic", "add_real_data", "stop"
    details: Dict = field(default_factory=dict)


class OverfittingDetector:
    """Detects overfitting to synthetic data artifacts.

    Compares performance on synthetic vs real data to detect
    when prompts are learning synthetic-specific patterns rather
    than generalizable PII detection.

    Example:
        detector = OverfittingDetector()

        report = detector.check_overfitting(
            flow_runner=run_flow,
            synthetic_samples=synthetic_docs,
            real_samples=real_docs,
        )

        if report.is_overfitting:
            print(f"Overfitting detected! Gap: {report.synthetic_real_gap}")
    """

    def __init__(
        self,
        gap_threshold: float = 0.15,  # 15% performance gap triggers alert
        metrics_to_check: Optional[List[str]] = None,
    ):
        """Initialize the overfitting detector.

        Args:
            gap_threshold: Maximum acceptable synthetic-real performance gap
            metrics_to_check: Which metrics to check (default: F1, precision, recall)
        """
        self.gap_threshold = gap_threshold
        self.metrics_to_check = metrics_to_check or [
            "exact_f1", "exact_precision", "exact_recall"
        ]

    def check_overfitting(
        self,
        flow_runner: Callable[[str], Dict],
        synthetic_samples: List[Dict],
        real_samples: List[Dict],
    ) -> OverfittingReport:
        """Check for overfitting by comparing synthetic vs real performance.

        Args:
            flow_runner: Function that runs the flow and returns final state dict
            synthetic_samples: Synthetic test samples with 'text' and 'entities'
            real_samples: Real test samples with 'text' and 'entities'

        Returns:
            OverfittingReport with analysis
        """
        # Evaluate on synthetic
        synthetic_metrics = self._evaluate(flow_runner, synthetic_samples)

        # Evaluate on real
        real_metrics = self._evaluate(flow_runner, real_samples)

        # Calculate gaps
        gap = {}
        for metric in self.metrics_to_check:
            synth_val = getattr(synthetic_metrics, metric, 0.0)
            real_val = getattr(real_metrics, metric, 0.0)
            gap[metric] = synth_val - real_val

        # Determine if overfitting
        is_overfitting = any(g > self.gap_threshold for g in gap.values())

        # Generate recommendation
        max_gap = max(gap.values()) if gap else 0
        if max_gap > self.gap_threshold * 2:
            recommendation = "stop"  # Severe overfitting
        elif max_gap > self.gap_threshold:
            recommendation = "diversify_synthetic"
        elif max_gap > self.gap_threshold * 0.5:
            recommendation = "add_real_data"
        else:
            recommendation = "continue"

        return OverfittingReport(
            is_overfitting=is_overfitting,
            synthetic_real_gap=gap,
            gap_threshold=self.gap_threshold,
            recommendation=recommendation,
            details={
                "synthetic_metrics": {
                    m: getattr(synthetic_metrics, m, 0.0)
                    for m in self.metrics_to_check
                },
                "real_metrics": {
                    m: getattr(real_metrics, m, 0.0)
                    for m in self.metrics_to_check
                },
                "max_gap": max_gap,
            },
        )

    def _evaluate(
        self,
        flow_runner: Callable,
        samples: List[Dict],
    ) -> MultiDimensionalMetrics:
        """Evaluate flow on samples."""
        expected_all = []
        detected_all = []

        for sample in samples:
            text = sample.get("text", "")
            expected = sample.get("entities", [])

            try:
                result = flow_runner(text)
                detected = result.get("detected_entities", [])
            except Exception:
                detected = []

            expected_all.append(expected)
            detected_all.append(detected)

        return MultiDimensionalMetrics.calculate(
            expected_entities=expected_all,
            detected_entities=detected_all,
        )

    def check_prompt_overfitting(
        self,
        prompt_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> bool:
        """Check if a specific prompt is overfitting.

        Args:
            prompt_metrics: Metrics on synthetic data
            baseline_metrics: Metrics on real/holdout data

        Returns:
            True if overfitting detected
        """
        for metric in self.metrics_to_check:
            synth = prompt_metrics.get(metric, 0.0)
            real = baseline_metrics.get(metric, 0.0)
            if synth - real > self.gap_threshold:
                return True
        return False

    def track_overfitting_trend(
        self,
        history: List[Tuple[Dict[str, float], Dict[str, float]]],
    ) -> Dict:
        """Track overfitting trend over time.

        Args:
            history: List of (synthetic_metrics, real_metrics) tuples

        Returns:
            Trend analysis
        """
        if len(history) < 2:
            return {"trend": "insufficient_data"}

        gaps = []
        for synth, real in history:
            gap = {}
            for metric in self.metrics_to_check:
                gap[metric] = synth.get(metric, 0) - real.get(metric, 0)
            gaps.append(gap)

        # Calculate trend
        f1_gaps = [g.get("exact_f1", 0) for g in gaps]
        trend_slope = (f1_gaps[-1] - f1_gaps[0]) / len(f1_gaps)

        if trend_slope > 0.02:
            trend = "increasing_overfit"
        elif trend_slope < -0.02:
            trend = "decreasing_overfit"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": trend_slope,
            "current_gap": gaps[-1],
            "initial_gap": gaps[0],
        }


class EarlyStoppingMonitor:
    """Monitors training to detect when to stop.

    Implements early stopping based on validation performance
    and overfitting detection.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        overfitting_detector: Optional[OverfittingDetector] = None,
    ):
        """Initialize early stopping monitor.

        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement to reset patience
            overfitting_detector: Optional overfitting detector
        """
        self.patience = patience
        self.min_delta = min_delta
        self.overfitting_detector = overfitting_detector

        self.best_score = -float("inf")
        self.wait = 0
        self.stopped = False
        self.stop_reason = None

    def check(
        self,
        current_score: float,
        synthetic_metrics: Optional[Dict[str, float]] = None,
        real_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current validation score
            synthetic_metrics: Optional synthetic data metrics
            real_metrics: Optional real data metrics

        Returns:
            True if should stop training
        """
        if self.stopped:
            return True

        # Check for improvement
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

        # Check patience
        if self.wait >= self.patience:
            self.stopped = True
            self.stop_reason = "no_improvement"
            return True

        # Check for overfitting
        if (self.overfitting_detector and
            synthetic_metrics is not None and
            real_metrics is not None):

            is_overfitting = self.overfitting_detector.check_prompt_overfitting(
                synthetic_metrics, real_metrics
            )
            if is_overfitting:
                self.stopped = True
                self.stop_reason = "overfitting"
                return True

        return False

    def reset(self) -> None:
        """Reset the monitor."""
        self.best_score = -float("inf")
        self.wait = 0
        self.stopped = False
        self.stop_reason = None
