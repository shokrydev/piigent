"""Multi-Dimensional Evaluation Metrics.

Goes beyond simple P/R/F1 to provide comprehensive evaluation including:
- Exact and partial span matching
- Type-only matching (ignoring boundaries)
- Boundary error analysis
- Confidence calibration (ECE)
- Per-type confusion matrices
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math


@dataclass
class MultiDimensionalMetrics:
    """Comprehensive evaluation metrics for NER/PII detection.

    Provides multiple views into model performance:
    - Span matching: exact vs partial overlap
    - Type matching: correct type regardless of boundaries
    - Boundary errors: off-by-one and partial span analysis
    - Leakage: PII that survives anonymization
    - Calibration: confidence score reliability (ECE)
    """

    # Exact span matching (same start, end, and type)
    exact_precision: float = 0.0
    exact_recall: float = 0.0
    exact_f1: float = 0.0

    # Partial span matching (50%+ overlap with correct type)
    partial_precision: float = 0.0
    partial_recall: float = 0.0
    partial_f1: float = 0.0

    # Type matching (correct type, ignoring boundaries)
    type_precision: float = 0.0
    type_recall: float = 0.0
    type_f1: float = 0.0

    # Boundary error analysis
    boundary_errors: int = 0
    partial_span_rate: float = 0.0  # Correct type, wrong boundaries

    # Leakage analysis
    leakage_rate: float = 0.0
    leakage_severity: float = 0.0  # Weighted by entity sensitivity

    # Confidence calibration
    ece: float = 0.0  # Expected Calibration Error
    overconfidence_rate: float = 0.0

    # Per-type breakdown
    per_type_f1: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Counts for reference
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @classmethod
    def calculate(
        cls,
        expected_entities: List[List[Dict]],
        detected_entities: List[List[Dict]],
        confidence_bins: int = 10,
        overlap_threshold: float = 0.5,
    ) -> "MultiDimensionalMetrics":
        """Calculate comprehensive metrics from expected vs detected entities.

        Args:
            expected_entities: List of expected entity lists (one per document)
            detected_entities: List of detected entity lists (one per document)
            confidence_bins: Number of bins for ECE calculation
            overlap_threshold: Minimum overlap ratio for partial match

        Returns:
            MultiDimensionalMetrics with all calculated values
        """
        metrics = cls()

        # Aggregate counts
        exact_tp, exact_fp, exact_fn = 0, 0, 0
        partial_tp, partial_fp, partial_fn = 0, 0, 0
        type_tp, type_fp, type_fn = 0, 0, 0

        boundary_errors = 0
        per_type_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # For ECE calculation
        confidence_correct: Dict[int, List[bool]] = defaultdict(list)
        confidence_values: Dict[int, List[float]] = defaultdict(list)

        for doc_expected, doc_detected in zip(expected_entities, detected_entities):
            # Convert to sets of tuples for matching
            expected_set = set()
            detected_set = set()

            for e in doc_expected:
                # Handle both dicts and objects
                start = e.get("start") if isinstance(e, dict) else getattr(e, "start", 0)
                end = e.get("end") if isinstance(e, dict) else getattr(e, "end", 0)
                etype = e.get("entity_type", e.get("type")) if isinstance(e, dict) else getattr(e, "entity_type", "UNKNOWN")
                expected_set.add((start, end, etype))

            for d in doc_detected:
                start = d.get("start") if isinstance(d, dict) else getattr(d, "start", 0)
                end = d.get("end") if isinstance(d, dict) else getattr(d, "end", 0)
                etype = d.get("entity_type", d.get("type")) if isinstance(d, dict) else getattr(d, "entity_type", "UNKNOWN")
                score = d.get("score", 0.85) if isinstance(d, dict) else getattr(d, "score", 0.85)
                
                detected_set.add((start, end, etype))

                # Track confidence for ECE
                bin_idx = min(int(score * confidence_bins), confidence_bins - 1)

                # Check if this detection is correct (exact match)
                is_correct = (start, end, etype) in expected_set
                confidence_correct[bin_idx].append(is_correct)
                confidence_values[bin_idx].append(score)

            # Exact matching
            doc_exact_tp = len(expected_set & detected_set)
            doc_exact_fp = len(detected_set - expected_set)
            doc_exact_fn = len(expected_set - detected_set)

            exact_tp += doc_exact_tp
            exact_fp += doc_exact_fp
            exact_fn += doc_exact_fn

            # Partial and type matching
            matched_expected = set()
            matched_detected = set()

            for exp in doc_expected:
                exp_start = exp.get("start") if isinstance(exp, dict) else getattr(exp, "start", 0)
                exp_end = exp.get("end") if isinstance(exp, dict) else getattr(exp, "end", 0)
                exp_type = exp.get("entity_type", exp.get("type")) if isinstance(exp, dict) else getattr(exp, "entity_type", "UNKNOWN")

                for det in doc_detected:
                    det_start = det.get("start") if isinstance(det, dict) else getattr(det, "start", 0)
                    det_end = det.get("end") if isinstance(det, dict) else getattr(det, "end", 0)
                    det_type = det.get("entity_type", det.get("type")) if isinstance(det, dict) else getattr(det, "entity_type", "UNKNOWN")

                    # Calculate overlap
                    overlap_start = max(exp_start, det_start)
                    overlap_end = min(exp_end, det_end)
                    overlap_len = max(0, overlap_end - overlap_start)

                    exp_len = exp_end - exp_start
                    det_len = det_end - det_start

                    overlap_ratio = overlap_len / max(exp_len, det_len) if max(exp_len, det_len) > 0 else 0

                    # Type match (any overlap)
                    if overlap_ratio > 0 and exp_type == det_type:
                        matched_expected.add((exp_start, exp_end, exp_type))
                        matched_detected.add((det_start, det_end, det_type))

                        # Check for boundary error
                        if (exp_start, exp_end) != (det_start, det_end):
                            boundary_errors += 1

                    # Partial match (50%+ overlap and correct type)
                    if overlap_ratio >= overlap_threshold and exp_type == det_type:
                        partial_tp += 1
                        per_type_stats[exp_type]["tp"] += 1

                    # Build confusion matrix
                    if overlap_ratio > 0:
                        confusion[det_type][exp_type] += 1

            # Type matching counts
            type_tp += len(matched_expected)
            
            # Helper to get attributes for comparison
            def _get_entity_key(item):
                 if isinstance(item, dict):
                     return (item.get("start"), item.get("end"), item.get("entity_type", item.get("type")))
                 return (item.start, item.end, item.entity_type)

            type_fn += len(set(_get_entity_key(e) for e in doc_expected) - matched_expected)
            type_fp += len(set(_get_entity_key(d) for d in doc_detected) - matched_detected)

            # Partial FP/FN (approximate)
            partial_fn += len(doc_expected) - len([e for e in doc_expected
                                                   if any(_overlap_ratio(e, d) >= overlap_threshold
                                                          for d in doc_detected)])

            # Track per-type FN
            for exp in doc_expected:
                exp_type = exp.get("entity_type", exp.get("type"))
                if not any(_overlap_ratio(exp, d) >= overlap_threshold and
                          d.get("entity_type", d.get("type")) == exp_type
                          for d in doc_detected):
                    per_type_stats[exp_type]["fn"] += 1

            # Track per-type FP
            for det in doc_detected:
                det_type = det.get("entity_type", det.get("type"))
                if not any(_overlap_ratio(exp, det) >= overlap_threshold and
                          exp.get("entity_type", exp.get("type")) == det_type
                          for exp in doc_expected):
                    per_type_stats[det_type]["fp"] += 1

        # Calculate final metrics
        metrics.exact_precision = _safe_div(exact_tp, exact_tp + exact_fp)
        metrics.exact_recall = _safe_div(exact_tp, exact_tp + exact_fn)
        metrics.exact_f1 = _f1(metrics.exact_precision, metrics.exact_recall)

        metrics.partial_precision = _safe_div(partial_tp, partial_tp + exact_fp)
        metrics.partial_recall = _safe_div(partial_tp, partial_tp + partial_fn)
        metrics.partial_f1 = _f1(metrics.partial_precision, metrics.partial_recall)

        metrics.type_precision = _safe_div(type_tp, type_tp + type_fp)
        metrics.type_recall = _safe_div(type_tp, type_tp + type_fn)
        metrics.type_f1 = _f1(metrics.type_precision, metrics.type_recall)

        metrics.boundary_errors = boundary_errors
        metrics.partial_span_rate = _safe_div(boundary_errors, type_tp) if type_tp > 0 else 0.0

        metrics.true_positives = exact_tp
        metrics.false_positives = exact_fp
        metrics.false_negatives = exact_fn

        # Per-type F1 scores
        for entity_type, stats in per_type_stats.items():
            p = _safe_div(stats["tp"], stats["tp"] + stats["fp"])
            r = _safe_div(stats["tp"], stats["tp"] + stats["fn"])
            metrics.per_type_f1[entity_type] = _f1(p, r)

        # Confusion matrix
        metrics.confusion_matrix = {k: dict(v) for k, v in confusion.items()}

        # ECE calculation
        ece = 0.0
        total_samples = 0
        overconfident = 0

        for bin_idx in range(confidence_bins):
            if bin_idx in confidence_correct:
                bin_accuracy = sum(confidence_correct[bin_idx]) / len(confidence_correct[bin_idx])
                bin_confidence = sum(confidence_values[bin_idx]) / len(confidence_values[bin_idx])
                bin_size = len(confidence_correct[bin_idx])

                ece += bin_size * abs(bin_accuracy - bin_confidence)
                total_samples += bin_size

                if bin_confidence > bin_accuracy:
                    overconfident += sum(1 for c, a in zip(confidence_values[bin_idx],
                                                          confidence_correct[bin_idx])
                                        if c > 0.7 and not a)

        metrics.ece = ece / total_samples if total_samples > 0 else 0.0
        metrics.overconfidence_rate = overconfident / total_samples if total_samples > 0 else 0.0

        return metrics

    def to_fitness_dict(self) -> Dict[str, float]:
        """Convert metrics to fitness scores dict for genome storage.

        Returns:
            Dict mapping metric names to scores
        """
        fitness = {
            "exact_f1": self.exact_f1,
            "exact_precision": self.exact_precision,
            "exact_recall": self.exact_recall,
            "partial_f1": self.partial_f1,
            "type_f1": self.type_f1,
            "ece": self.ece,
        }

        # Add per-type F1 scores
        for entity_type, f1 in self.per_type_f1.items():
            fitness[f"f1_{entity_type.lower()}"] = f1

        return fitness

    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        lines = [
            "=== Multi-Dimensional Evaluation ===",
            f"Exact Match: P={self.exact_precision:.3f} R={self.exact_recall:.3f} F1={self.exact_f1:.3f}",
            f"Partial Match: P={self.partial_precision:.3f} R={self.partial_recall:.3f} F1={self.partial_f1:.3f}",
            f"Type Match: P={self.type_precision:.3f} R={self.type_recall:.3f} F1={self.type_f1:.3f}",
            f"Boundary Errors: {self.boundary_errors} ({self.partial_span_rate:.1%} of type matches)",
            f"Calibration (ECE): {self.ece:.3f}, Overconfidence: {self.overconfidence_rate:.1%}",
            "",
            "Per-Type F1:",
        ]

        for entity_type, f1 in sorted(self.per_type_f1.items(), key=lambda x: -x[1]):
            lines.append(f"  {entity_type}: {f1:.3f}")

        return "\n".join(lines)


def _safe_div(num: float, denom: float) -> float:
    """Safe division returning 0 when denominator is 0."""
    return num / denom if denom > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _overlap_ratio(e1: Dict, e2: Dict) -> float:
    """Calculate overlap ratio between two entities."""
    s1, e1_end = e1.get("start"), e1.get("end")
    s2, e2_end = e2.get("start"), e2.get("end")

    overlap_start = max(s1, s2)
    overlap_end = min(e1_end, e2_end)
    overlap_len = max(0, overlap_end - overlap_start)

    max_len = max(e1_end - s1, e2_end - s2)
    return overlap_len / max_len if max_len > 0 else 0.0
