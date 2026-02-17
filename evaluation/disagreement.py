"""Disagreement-Based Evaluation.

Multiple recognizers annotate the same sample; disagreement indicates
weakness in the detection system.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple


class DisagreementType(str, Enum):
    """Types of disagreement between recognizers."""
    TYPE_CONFLICT = "type_conflict"       # Same span, different types
    PRESENCE_CONFLICT = "presence"        # One detected, other didn't
    BOUNDARY_CONFLICT = "boundary"        # Overlapping with different boundaries
    CONFIDENCE_GAP = "confidence_gap"     # Same detection, large confidence gap


@dataclass
class DisagreementCase:
    """A case where recognizers disagree."""
    text: str
    span: Tuple[int, int]  # (start, end)
    disagreement_type: DisagreementType
    predictions: Dict[str, Dict]  # recognizer_name -> prediction
    context: str = ""
    severity: float = 0.0  # 0-1, higher = more severe disagreement


@dataclass
class DisagreementReport:
    """Report of disagreements found."""
    total_documents: int
    total_disagreements: int
    by_type: Dict[str, int]
    by_entity_type: Dict[str, int]
    cases: List[DisagreementCase]
    agreement_rate: float  # 0-1


class DisagreementEvaluator:
    """Evaluates disagreement between multiple recognizers.

    Multiple recognizers annotate the same samples; disagreement
    indicates weakness that needs attention.

    Example:
        evaluator = DisagreementEvaluator([
            ("pattern", pattern_recognizer),
            ("llm", llm_recognizer),
            ("ner", ner_recognizer),
        ])

        disagreements = evaluator.find_disagreements(document)
        for case in disagreements:
            print(f"Disagreement at {case.span}: {case.predictions}")
    """

    def __init__(
        self,
        recognizers: List[Tuple[str, Callable]],
        confidence_gap_threshold: float = 0.3,
    ):
        """Initialize the disagreement evaluator.

        Args:
            recognizers: List of (name, recognizer_function) tuples
            confidence_gap_threshold: Minimum confidence gap to flag
        """
        self.recognizers = recognizers
        self.confidence_gap_threshold = confidence_gap_threshold

    def find_disagreements(
        self,
        document: str,
    ) -> List[DisagreementCase]:
        """Find cases where recognizers disagree.

        Args:
            document: Text to analyze

        Returns:
            List of disagreement cases
        """
        # Run all recognizers
        all_results = {}
        for name, recognizer in self.recognizers:
            try:
                results = recognizer(document)
                all_results[name] = results if isinstance(results, list) else []
            except Exception as e:
                all_results[name] = []

        # Find all unique spans
        all_spans = self._collect_all_spans(all_results)

        # Check for disagreements at each span
        disagreements = []
        for span in all_spans:
            predictions = self._get_predictions_for_span(span, all_results)
            disagreement = self._check_disagreement(span, predictions, document)
            if disagreement:
                disagreements.append(disagreement)

        return disagreements

    def _collect_all_spans(
        self,
        all_results: Dict[str, List[Dict]],
    ) -> Set[Tuple[int, int]]:
        """Collect all unique spans from all recognizers."""
        spans = set()
        for results in all_results.values():
            for entity in results:
                spans.add((entity.get("start", 0), entity.get("end", 0)))
        return spans

    def _get_predictions_for_span(
        self,
        span: Tuple[int, int],
        all_results: Dict[str, List[Dict]],
    ) -> Dict[str, Optional[Dict]]:
        """Get predictions from each recognizer for a span."""
        predictions = {}

        for recognizer_name, results in all_results.items():
            # Find exact or overlapping match
            match = None
            for entity in results:
                entity_span = (entity.get("start", 0), entity.get("end", 0))
                if self._spans_overlap(span, entity_span):
                    match = entity
                    break
            predictions[recognizer_name] = match

        return predictions

    def _spans_overlap(
        self,
        span1: Tuple[int, int],
        span2: Tuple[int, int],
    ) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _check_disagreement(
        self,
        span: Tuple[int, int],
        predictions: Dict[str, Optional[Dict]],
        document: str,
    ) -> Optional[DisagreementCase]:
        """Check if predictions disagree."""
        # Filter to recognizers that made predictions
        has_prediction = {k: v for k, v in predictions.items() if v is not None}
        no_prediction = {k for k, v in predictions.items() if v is None}

        # No disagreement if all agree (all detected or none detected)
        if not has_prediction or not no_prediction:
            # Check for type disagreement among those with predictions
            if len(has_prediction) < 2:
                return None

            types = {
                k: v.get("entity_type", v.get("type", ""))
                for k, v in has_prediction.items()
            }
            unique_types = set(types.values())

            if len(unique_types) == 1:
                # All agree on type, check confidence gap
                confidences = [
                    v.get("score", 0.5) for v in has_prediction.values()
                ]
                if max(confidences) - min(confidences) >= self.confidence_gap_threshold:
                    return DisagreementCase(
                        text=document[span[0]:span[1]],
                        span=span,
                        disagreement_type=DisagreementType.CONFIDENCE_GAP,
                        predictions=predictions,
                        context=document[max(0, span[0]-30):min(len(document), span[1]+30)],
                        severity=0.3,
                    )
                return None

            # Type conflict
            return DisagreementCase(
                text=document[span[0]:span[1]],
                span=span,
                disagreement_type=DisagreementType.TYPE_CONFLICT,
                predictions=predictions,
                context=document[max(0, span[0]-30):min(len(document), span[1]+30)],
                severity=0.7,
            )

        # Presence conflict - some detected, some didn't
        return DisagreementCase(
            text=document[span[0]:span[1]],
            span=span,
            disagreement_type=DisagreementType.PRESENCE_CONFLICT,
            predictions=predictions,
            context=document[max(0, span[0]-30):min(len(document), span[1]+30)],
            severity=0.8,
        )

    def evaluate_documents(
        self,
        documents: List[str],
    ) -> DisagreementReport:
        """Evaluate multiple documents for disagreements.

        Args:
            documents: List of documents to evaluate

        Returns:
            DisagreementReport with statistics
        """
        all_cases = []
        total_predictions = 0
        agreements = 0

        for doc in documents:
            cases = self.find_disagreements(doc)
            all_cases.extend(cases)

            # Count total predictions for agreement rate
            for name, recognizer in self.recognizers:
                try:
                    results = recognizer(doc)
                    total_predictions += len(results) if isinstance(results, list) else 0
                except Exception:
                    pass

        # Calculate statistics
        by_type = defaultdict(int)
        by_entity_type = defaultdict(int)

        for case in all_cases:
            by_type[case.disagreement_type.value] += 1
            for pred in case.predictions.values():
                if pred:
                    entity_type = pred.get("entity_type", pred.get("type", ""))
                    if entity_type:
                        by_entity_type[entity_type] += 1

        # Agreement rate (1 - disagreement rate)
        total_spans = len(all_cases) + (total_predictions - len(all_cases) * len(self.recognizers))
        agreement_rate = 1 - (len(all_cases) / max(total_spans, 1))

        return DisagreementReport(
            total_documents=len(documents),
            total_disagreements=len(all_cases),
            by_type=dict(by_type),
            by_entity_type=dict(by_entity_type),
            cases=all_cases,
            agreement_rate=agreement_rate,
        )

    def get_high_disagreement_spans(
        self,
        documents: List[str],
        min_severity: float = 0.5,
    ) -> List[DisagreementCase]:
        """Get spans with high disagreement for human review.

        Args:
            documents: Documents to analyze
            min_severity: Minimum severity threshold

        Returns:
            High-severity disagreement cases
        """
        report = self.evaluate_documents(documents)
        return [c for c in report.cases if c.severity >= min_severity]

    def suggest_ground_truth(
        self,
        case: DisagreementCase,
        strategy: str = "majority",
    ) -> Optional[Dict]:
        """Suggest ground truth for a disagreement case.

        Args:
            case: The disagreement case
            strategy: Resolution strategy ("majority", "highest_confidence", "most_specific")

        Returns:
            Suggested entity dict or None
        """
        valid_predictions = [
            (name, pred) for name, pred in case.predictions.items()
            if pred is not None
        ]

        if not valid_predictions:
            return None

        if strategy == "majority":
            # Count type votes
            type_votes = defaultdict(list)
            for name, pred in valid_predictions:
                entity_type = pred.get("entity_type", pred.get("type", ""))
                type_votes[entity_type].append(pred)

            # Return prediction with most votes
            best_type = max(type_votes.keys(), key=lambda t: len(type_votes[t]))
            return type_votes[best_type][0]

        elif strategy == "highest_confidence":
            return max(
                [p for _, p in valid_predictions],
                key=lambda p: p.get("score", 0)
            )

        elif strategy == "most_specific":
            # Use type specificity
            from components.reflective_resolution import TYPE_SPECIFICITY
            return max(
                [p for _, p in valid_predictions],
                key=lambda p: TYPE_SPECIFICITY.get(
                    p.get("entity_type", p.get("type", "")), 0
                )
            )

        return valid_predictions[0][1] if valid_predictions else None
