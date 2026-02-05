"""Evaluation metrics for PII detection benchmarking.

Calculates precision, recall, F1 comparing detected entities against ground truth.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from graph.state import DetectedEntity


@dataclass
class MatchResult:
    """Result of matching detected entities to ground truth."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def spans_overlap(start1: int, end1: int, start2: int, end2: int, threshold: float = 0.5) -> bool:
    """Check if two spans overlap by at least threshold proportion."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_len = max(0, overlap_end - overlap_start)

    len1 = end1 - start1
    len2 = end2 - start2

    if len1 == 0 or len2 == 0:
        return False

    # Overlap proportion relative to smaller span
    min_len = min(len1, len2)
    return overlap_len / min_len >= threshold


def match_entities(
    detected: List[DetectedEntity],
    ground_truth: List[dict],
    strict_type_match: bool = True,
    overlap_threshold: float = 0.5,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Match detected entities to ground truth annotations.

    Args:
        detected: List of detected entities from flow.
        ground_truth: List of annotation dicts with entity_type, start, end.
        strict_type_match: If True, entity types must match exactly.
        overlap_threshold: Minimum overlap proportion for a match.

    Returns:
        Tuple of (matched_detected_indices, matched_gt_indices, unmatched_gt_indices)
    """
    matched_detected = set()
    matched_gt = set()

    for d_idx, det in enumerate(detected):
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            # Check type match
            if strict_type_match:
                # Allow some type mappings (e.g., PERSON matches PERSON)
                type_match = det.entity_type == gt["entity_type"]
            else:
                type_match = True

            # Check span overlap
            if type_match and spans_overlap(
                det.start, det.end,
                gt["start"], gt["end"],
                overlap_threshold,
            ):
                matched_detected.add(d_idx)
                matched_gt.add(gt_idx)
                break

    unmatched_gt = set(range(len(ground_truth))) - matched_gt

    return matched_detected, matched_gt, unmatched_gt


def calculate_metrics(
    detected: List[DetectedEntity],
    ground_truth: List[dict],
    strict_type_match: bool = True,
) -> MatchResult:
    """Calculate precision, recall, F1 for a single document.

    Args:
        detected: Detected entities from flow.
        ground_truth: Ground truth annotations.
        strict_type_match: Require entity types to match.

    Returns:
        MatchResult with metrics.
    """
    matched_det, matched_gt, unmatched_gt = match_entities(
        detected, ground_truth, strict_type_match
    )

    tp = len(matched_gt)
    fp = len(detected) - len(matched_det)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MatchResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_dataset_metrics(
    results: List[Tuple[List[DetectedEntity], List[dict]]],
    strict_type_match: bool = True,
) -> Dict:
    """Calculate aggregate metrics across a dataset.

    Args:
        results: List of (detected_entities, ground_truth) tuples.
        strict_type_match: Require entity types to match.

    Returns:
        Dict with aggregate metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_type_stats = {}

    for detected, ground_truth in results:
        # Overall metrics
        match = calculate_metrics(detected, ground_truth, strict_type_match)
        total_tp += match.true_positives
        total_fp += match.false_positives
        total_fn += match.false_negatives

        # Per-type metrics
        for gt in ground_truth:
            entity_type = gt["entity_type"]
            if entity_type not in per_type_stats:
                per_type_stats[entity_type] = {"tp": 0, "fp": 0, "fn": 0}

            # Check if this GT was matched
            gt_detected = any(
                d.entity_type == entity_type and
                spans_overlap(d.start, d.end, gt["start"], gt["end"])
                for d in detected
            )

            if gt_detected:
                per_type_stats[entity_type]["tp"] += 1
            else:
                per_type_stats[entity_type]["fn"] += 1

        # Count FPs per type
        for det in detected:
            entity_type = det.entity_type
            if entity_type not in per_type_stats:
                per_type_stats[entity_type] = {"tp": 0, "fp": 0, "fn": 0}

            det_matched = any(
                gt["entity_type"] == entity_type and
                spans_overlap(det.start, det.end, gt["start"], gt["end"])
                for gt in ground_truth
            )

            if not det_matched:
                per_type_stats[entity_type]["fp"] += 1

    # Calculate aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-type metrics
    per_type_metrics = {}
    for entity_type, stats in per_type_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type_metrics[entity_type] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
            "support": tp + fn,
        }

    return {
        "overall": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
        "per_type": per_type_metrics,
    }
