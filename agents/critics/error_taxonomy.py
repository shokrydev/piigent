"""Error Taxonomy Agent.

Classifies detection errors into systematic categories
for pattern analysis and targeted improvements.
"""

import re
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ErrorType(str, Enum):
    """Systematic error categories."""
    # Boundary errors
    BOUNDARY_LEFT = "boundary_left"       # Started too early
    BOUNDARY_RIGHT = "boundary_right"     # Ended too late
    BOUNDARY_SHORT = "boundary_short"     # Started late or ended early

    # Type errors
    TYPE_CONFUSION = "type_confusion"     # Correct span, wrong type
    FALSE_POSITIVE = "false_positive"     # Detected non-entity
    FALSE_NEGATIVE = "false_negative"     # Missed real entity

    # Overlap errors
    PARTIAL_OVERLAP = "partial_overlap"   # Partial match only
    ABSORBED = "absorbed"                 # Entity absorbed by larger

    # Context errors
    CONTEXT_DEPENDENT = "context_dependent"  # Needs specific context
    STRUCTURAL_CONFUSION = "structural"      # Confused with structure

    # Format errors
    FORMAT_VARIATION = "format_variation"    # Non-standard format
    CASE_SENSITIVITY = "case_sensitivity"    # Case variation issue


@dataclass
class ClassifiedError:
    """A classified detection error."""
    error_type: ErrorType
    entity_text: str
    expected_type: Optional[str]
    detected_type: Optional[str]
    expected_span: Tuple[int, int]
    detected_span: Optional[Tuple[int, int]]
    context: str
    explanation: str
    severity: float = 1.0  # 0-1, higher = more severe
    metadata: Dict = field(default_factory=dict)


class ErrorTaxonomyAgent:
    """Agent that classifies errors into systematic categories.

    Analyzes detection errors to identify patterns and root causes,
    enabling targeted improvements to the detection flow.

    Example:
        agent = ErrorTaxonomyAgent()

        error = agent.classify_error(
            expected={"text": "Ingenieur", "type": "OCCUPATION", "start": 10, "end": 19},
            detected={"text": "Ingenieur", "type": "PERSON", "start": 10, "end": 19, "score": 0.7},
            context="arbeitet als Ingenieur in München",
        )

        print(f"{error.error_type}: {error.explanation}")
    """

    # Type confusion pairs (commonly confused types)
    CONFUSION_PAIRS = {
        ("PERSON", "ORGANIZATION"): "Name vs institution",
        ("LOCATION", "ORGANIZATION"): "Place vs organization",
        ("OCCUPATION", "PERSON"): "Job vs name",
        ("AGE", "DATE_TIME"): "Age vs date",
        ("DE_POSTAL_CODE", "LOCATION"): "Postal code vs location",
    }

    # Severity weights by error type
    SEVERITY_WEIGHTS = {
        ErrorType.FALSE_NEGATIVE: 1.0,       # Worst - missed PII
        ErrorType.TYPE_CONFUSION: 0.8,
        ErrorType.PARTIAL_OVERLAP: 0.6,
        ErrorType.BOUNDARY_LEFT: 0.4,
        ErrorType.BOUNDARY_RIGHT: 0.4,
        ErrorType.BOUNDARY_SHORT: 0.5,
        ErrorType.FALSE_POSITIVE: 0.3,
        ErrorType.ABSORBED: 0.7,
        ErrorType.CONTEXT_DEPENDENT: 0.6,
        ErrorType.FORMAT_VARIATION: 0.5,
        ErrorType.CASE_SENSITIVITY: 0.3,
        ErrorType.STRUCTURAL_CONFUSION: 0.4,
    }

    def __init__(self):
        """Initialize the error taxonomy agent."""
        self.classified_errors: List[ClassifiedError] = []

    def classify_error(
        self,
        expected: Dict,
        detected: Optional[Dict],
        context: str,
    ) -> ClassifiedError:
        """Classify a detection error.

        Args:
            expected: Expected entity (ground truth)
            detected: Detected entity (or None if missed)
            context: Surrounding context text

        Returns:
            ClassifiedError with type and explanation
        """
        expected_text = expected.get("text", "")
        expected_type = expected.get("entity_type", expected.get("type", ""))
        expected_span = (expected["start"], expected["end"])

        # False negative (missed entirely)
        if detected is None:
            return self._classify_miss(expected, context)

        detected_text = detected.get("text", "")
        detected_type = detected.get("entity_type", detected.get("type", ""))
        detected_span = (detected["start"], detected["end"])

        # Exact match is not an error
        if expected_span == detected_span and expected_type == detected_type:
            # This shouldn't happen - it's a correct detection
            pass

        # Type confusion (same span, different type)
        if expected_span == detected_span and expected_type != detected_type:
            return self._classify_type_confusion(expected, detected, context)

        # Boundary errors
        if expected_type == detected_type:
            overlap = self._calculate_overlap(expected_span, detected_span)
            if overlap > 0:
                return self._classify_boundary_error(expected, detected, context)

        # Partial overlap with different type
        overlap = self._calculate_overlap(expected_span, detected_span)
        if overlap > 0:
            return ClassifiedError(
                error_type=ErrorType.PARTIAL_OVERLAP,
                entity_text=expected_text,
                expected_type=expected_type,
                detected_type=detected_type,
                expected_span=expected_span,
                detected_span=detected_span,
                context=context,
                explanation=f"Partial overlap ({overlap:.0%}) with different type: "
                           f"expected {expected_type}, got {detected_type}",
                severity=self.SEVERITY_WEIGHTS[ErrorType.PARTIAL_OVERLAP],
            )

        # No overlap - treat as false negative
        return self._classify_miss(expected, context)

    def _classify_miss(
        self,
        expected: Dict,
        context: str,
    ) -> ClassifiedError:
        """Classify a missed entity (false negative)."""
        expected_text = expected.get("text", "")
        expected_type = expected.get("entity_type", expected.get("type", ""))
        expected_span = (expected["start"], expected["end"])

        # Analyze why it might have been missed
        error_type = ErrorType.FALSE_NEGATIVE
        explanation = f"Missed {expected_type}: '{expected_text}'"

        # Check for context dependency
        if self._needs_context(expected_text, expected_type, context):
            error_type = ErrorType.CONTEXT_DEPENDENT
            explanation += " - requires specific context words"

        # Check for format variation
        if self._is_format_variation(expected_text, expected_type):
            error_type = ErrorType.FORMAT_VARIATION
            explanation += " - non-standard format"

        # Check for structural confusion
        structural_patterns = [":", "Datum", "Nr.", "Tel", "Fax"]
        if any(p in context for p in structural_patterns):
            error_type = ErrorType.STRUCTURAL_CONFUSION
            explanation += " - may be confused with document structure"

        return ClassifiedError(
            error_type=error_type,
            entity_text=expected_text,
            expected_type=expected_type,
            detected_type=None,
            expected_span=expected_span,
            detected_span=None,
            context=context,
            explanation=explanation,
            severity=self.SEVERITY_WEIGHTS[error_type],
        )

    def _classify_type_confusion(
        self,
        expected: Dict,
        detected: Dict,
        context: str,
    ) -> ClassifiedError:
        """Classify a type confusion error."""
        expected_type = expected.get("entity_type", expected.get("type", ""))
        detected_type = detected.get("entity_type", detected.get("type", ""))
        expected_text = expected.get("text", "")
        expected_span = (expected["start"], expected["end"])
        detected_span = (detected["start"], detected["end"])

        # Check for known confusion pairs
        pair = (expected_type, detected_type)
        reverse_pair = (detected_type, expected_type)

        if pair in self.CONFUSION_PAIRS:
            explanation = f"Common confusion: {self.CONFUSION_PAIRS[pair]}"
        elif reverse_pair in self.CONFUSION_PAIRS:
            explanation = f"Common confusion: {self.CONFUSION_PAIRS[reverse_pair]}"
        else:
            explanation = f"Type confusion: expected {expected_type}, detected {detected_type}"

        return ClassifiedError(
            error_type=ErrorType.TYPE_CONFUSION,
            entity_text=expected_text,
            expected_type=expected_type,
            detected_type=detected_type,
            expected_span=expected_span,
            detected_span=detected_span,
            context=context,
            explanation=explanation,
            severity=self.SEVERITY_WEIGHTS[ErrorType.TYPE_CONFUSION],
            metadata={"confusion_pair": (expected_type, detected_type)},
        )

    def _classify_boundary_error(
        self,
        expected: Dict,
        detected: Dict,
        context: str,
    ) -> ClassifiedError:
        """Classify a boundary error."""
        expected_text = expected.get("text", "")
        expected_type = expected.get("entity_type", expected.get("type", ""))
        expected_span = (expected["start"], expected["end"])
        detected_span = (detected["start"], detected["end"])

        start_diff = detected_span[0] - expected_span[0]
        end_diff = detected_span[1] - expected_span[1]

        if start_diff < 0:
            error_type = ErrorType.BOUNDARY_LEFT
            explanation = f"Started {-start_diff} chars early"
        elif end_diff > 0:
            error_type = ErrorType.BOUNDARY_RIGHT
            explanation = f"Ended {end_diff} chars late"
        elif start_diff > 0 or end_diff < 0:
            error_type = ErrorType.BOUNDARY_SHORT
            explanation = f"Span too short by {start_diff + (-end_diff)} chars"
        else:
            error_type = ErrorType.PARTIAL_OVERLAP
            explanation = "Boundary mismatch"

        return ClassifiedError(
            error_type=error_type,
            entity_text=expected_text,
            expected_type=expected_type,
            detected_type=expected_type,
            expected_span=expected_span,
            detected_span=detected_span,
            context=context,
            explanation=explanation,
            severity=self.SEVERITY_WEIGHTS[error_type],
            metadata={"start_diff": start_diff, "end_diff": end_diff},
        )

    def _calculate_overlap(
        self,
        span1: Tuple[int, int],
        span2: Tuple[int, int],
    ) -> float:
        """Calculate overlap ratio between two spans."""
        overlap_start = max(span1[0], span2[0])
        overlap_end = min(span1[1], span2[1])

        if overlap_end <= overlap_start:
            return 0.0

        overlap_len = overlap_end - overlap_start
        min_len = min(span1[1] - span1[0], span2[1] - span2[0])

        return overlap_len / min_len if min_len > 0 else 0.0

    def _needs_context(
        self,
        entity_text: str,
        entity_type: str,
        context: str,
    ) -> bool:
        """Check if entity detection requires specific context."""
        # Context keywords that help detection
        context_keywords = {
            "OCCUPATION": ["arbeitet", "beruflich", "tätig", "Beruf"],
            "AGE": ["Jahre", "jährig", "Alter", "geboren"],
            "PERSON": ["Herr", "Frau", "Dr.", "Patient"],
            "LOCATION": ["wohnhaft", "aus", "in", "nach"],
        }

        keywords = context_keywords.get(entity_type, [])
        return any(kw in context for kw in keywords)

    def _is_format_variation(
        self,
        entity_text: str,
        entity_type: str,
    ) -> bool:
        """Check if entity has non-standard format."""
        # Standard patterns
        patterns = {
            "DATE_TIME": r"\d{1,2}\.\d{1,2}\.\d{2,4}",
            "DE_POSTAL_CODE": r"\d{5}",
            "DE_KVNR": r"[A-Z]\d{9}",
        }

        if entity_type in patterns:
            return not re.fullmatch(patterns[entity_type], entity_text)

        return False

    def analyze_batch(
        self,
        expected: List[Dict],
        detected: List[Dict],
        text: str,
    ) -> List[ClassifiedError]:
        """Analyze a batch of expected vs detected entities.

        Args:
            expected: Expected entities
            detected: Detected entities
            text: Full document text

        Returns:
            List of classified errors
        """
        errors = []

        # Build detected lookup
        detected_by_span = {}
        for d in detected:
            span = (d["start"], d["end"])
            detected_by_span[span] = d

        # Analyze each expected entity
        for exp in expected:
            exp_span = (exp["start"], exp["end"])
            exp_type = exp.get("entity_type", exp.get("type", ""))

            # Find matching or overlapping detection
            matching_det = None

            # Check for exact span match
            if exp_span in detected_by_span:
                matching_det = detected_by_span[exp_span]
            else:
                # Find best overlapping detection
                best_overlap = 0
                for det in detected:
                    overlap = self._calculate_overlap(exp_span, (det["start"], det["end"]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        matching_det = det if overlap >= 0.5 else None

            # Skip if correct detection
            if matching_det:
                det_span = (matching_det["start"], matching_det["end"])
                det_type = matching_det.get("entity_type", matching_det.get("type", ""))
                if exp_span == det_span and exp_type == det_type:
                    continue

            # Extract context
            ctx_start = max(0, exp["start"] - 30)
            ctx_end = min(len(text), exp["end"] + 30)
            context = text[ctx_start:ctx_end]

            error = self.classify_error(exp, matching_det, context)
            errors.append(error)
            self.classified_errors.append(error)

        return errors

    def get_error_summary(self) -> Dict:
        """Get summary statistics of classified errors.

        Returns:
            Summary dict with counts and patterns
        """
        by_type = defaultdict(int)
        by_entity_type = defaultdict(int)
        confusion_pairs = defaultdict(int)

        for error in self.classified_errors:
            by_type[error.error_type.value] += 1
            if error.expected_type:
                by_entity_type[error.expected_type] += 1
            if error.error_type == ErrorType.TYPE_CONFUSION:
                pair = error.metadata.get("confusion_pair")
                if pair:
                    confusion_pairs[pair] += 1

        return {
            "total_errors": len(self.classified_errors),
            "by_error_type": dict(by_type),
            "by_entity_type": dict(by_entity_type),
            "confusion_pairs": dict(confusion_pairs),
            "avg_severity": (
                sum(e.severity for e in self.classified_errors) / len(self.classified_errors)
                if self.classified_errors else 0.0
            ),
        }

    def get_improvement_priorities(self) -> List[Dict]:
        """Get prioritized list of improvement areas.

        Returns:
            List of improvement priorities with rationale
        """
        summary = self.get_error_summary()
        priorities = []

        # Prioritize by error type frequency and severity
        for error_type, count in sorted(
            summary["by_error_type"].items(),
            key=lambda x: -x[1] * self.SEVERITY_WEIGHTS.get(ErrorType(x[0]), 0.5)
        ):
            priorities.append({
                "area": error_type,
                "count": count,
                "severity": self.SEVERITY_WEIGHTS.get(ErrorType(error_type), 0.5),
                "priority_score": count * self.SEVERITY_WEIGHTS.get(ErrorType(error_type), 0.5),
            })

        return priorities[:5]  # Top 5 priorities

    def reset(self) -> None:
        """Reset classified errors."""
        self.classified_errors = []
