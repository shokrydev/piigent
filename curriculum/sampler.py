"""Failure-Weighted Sampler.

Tracks failures and adjusts sampling weights to generate more
data around recent failures, enabling focused improvement.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class FailureRecord:
    """Record of a detection failure."""
    entity_type: str
    context_pattern: str  # Simplified context (e.g., "social_history", "signature")
    failure_type: str     # "missed", "wrong_type", "wrong_boundary"
    entity_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class FailureWeightedSampler:
    """Samples data with weights based on failure frequency.

    Tracks failures by entity type and context, then provides
    sampling weights that oversample failure-prone combinations.

    Example:
        sampler = FailureWeightedSampler()

        # Record failures during evaluation
        sampler.record_failure("OCCUPATION", "social_history", "missed", "Ingenieur")
        sampler.record_failure("OCCUPATION", "social_history", "missed", "Lehrerin")

        # Get weights for SynPII generation
        weights = sampler.get_sampling_weights()
        # {"OCCUPATION": 1.5, ...}  - OCCUPATION is oversampled
    """

    def __init__(
        self,
        decay_factor: float = 0.95,     # How fast old failures decay
        base_weight: float = 1.0,       # Default weight for entity types
        max_weight_multiplier: float = 3.0,  # Maximum weight boost
        min_failures_for_boost: int = 2,     # Minimum failures to trigger boost
    ):
        """Initialize the failure-weighted sampler.

        Args:
            decay_factor: Exponential decay for old failures (per evaluation)
            base_weight: Default weight for all entity types
            max_weight_multiplier: Maximum boost multiplier
            min_failures_for_boost: Minimum failures needed for boost
        """
        self.decay_factor = decay_factor
        self.base_weight = base_weight
        self.max_weight_multiplier = max_weight_multiplier
        self.min_failures_for_boost = min_failures_for_boost

        # Track failures
        self.failures: List[FailureRecord] = []
        self.failure_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Track patterns
        self.context_patterns: Dict[str, List[str]] = defaultdict(list)

    def record_failure(
        self,
        entity_type: str,
        context: str,
        failure_type: str,
        entity_text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a detection failure.

        Args:
            entity_type: The entity type that failed
            context: Context where failure occurred (e.g., "social_history")
            failure_type: Type of failure ("missed", "wrong_type", "wrong_boundary")
            entity_text: The actual entity text
            metadata: Additional metadata about the failure
        """
        record = FailureRecord(
            entity_type=entity_type,
            context_pattern=context,
            failure_type=failure_type,
            entity_text=entity_text,
            metadata=metadata or {},
        )
        self.failures.append(record)
        self.failure_counts[entity_type][failure_type] += 1
        self.context_patterns[entity_type].append(context)

    def record_batch_failures(
        self,
        expected: List[Dict],
        detected: List[Dict],
        text: str,
        context: str = "unknown",
    ) -> int:
        """Record failures from comparing expected vs detected entities.

        Args:
            expected: List of expected entities
            detected: List of detected entities
            text: Full document text
            context: Context identifier

        Returns:
            Number of failures recorded
        """
        failures_count = 0

        # Build detected lookup
        detected_spans = {
            (d["start"], d["end"], d.get("entity_type", d.get("type"))): d
            for d in detected
        }

        for exp in expected:
            exp_type = exp.get("entity_type", exp.get("type"))
            exp_text = exp.get("text", text[exp["start"]:exp["end"]])
            exp_key = (exp["start"], exp["end"], exp_type)

            if exp_key not in detected_spans:
                # Check if span was detected with wrong type
                span_detected = any(
                    d["start"] == exp["start"] and d["end"] == exp["end"]
                    for d in detected
                )

                if span_detected:
                    failure_type = "wrong_type"
                else:
                    # Check for partial overlap
                    partial = any(
                        self._overlaps(exp, d) for d in detected
                    )
                    failure_type = "wrong_boundary" if partial else "missed"

                self.record_failure(
                    entity_type=exp_type,
                    context=context,
                    failure_type=failure_type,
                    entity_text=exp_text,
                )
                failures_count += 1

        return failures_count

    def _overlaps(self, e1: Dict, e2: Dict) -> bool:
        """Check if two entities overlap."""
        return not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"])

    def get_sampling_weights(self) -> Dict[str, float]:
        """Get sampling weights based on failure frequency.

        Returns:
            Dict mapping entity types to sampling weights
        """
        weights = {}

        # Calculate total failures per type
        type_totals = {}
        for entity_type, type_failures in self.failure_counts.items():
            type_totals[entity_type] = sum(type_failures.values())

        if not type_totals:
            return weights

        # Normalize and calculate weights
        max_failures = max(type_totals.values())
        mean_failures = sum(type_totals.values()) / len(type_totals)

        for entity_type, failures in type_totals.items():
            if failures < self.min_failures_for_boost:
                weights[entity_type] = self.base_weight
            else:
                # Calculate boost based on failure rate relative to mean
                ratio = failures / mean_failures if mean_failures > 0 else 1.0
                boost = min(ratio, self.max_weight_multiplier)
                weights[entity_type] = self.base_weight * boost

        return weights

    def get_context_weights(self, entity_type: str) -> Dict[str, float]:
        """Get sampling weights for different contexts for an entity type.

        Args:
            entity_type: The entity type

        Returns:
            Dict mapping context patterns to weights
        """
        contexts = self.context_patterns.get(entity_type, [])
        if not contexts:
            return {}

        context_counts = defaultdict(int)
        for ctx in contexts:
            context_counts[ctx] += 1

        max_count = max(context_counts.values())
        return {
            ctx: count / max_count * self.max_weight_multiplier
            for ctx, count in context_counts.items()
        }

    def get_failure_analysis(self) -> Dict:
        """Get detailed failure analysis.

        Returns:
            Analysis dict with breakdowns by type, context, failure mode
        """
        by_type = defaultdict(int)
        by_failure_type = defaultdict(int)
        by_context = defaultdict(int)
        by_type_and_failure = defaultdict(lambda: defaultdict(int))

        for failure in self.failures:
            by_type[failure.entity_type] += 1
            by_failure_type[failure.failure_type] += 1
            by_context[failure.context_pattern] += 1
            by_type_and_failure[failure.entity_type][failure.failure_type] += 1

        # Find most common failure patterns
        entity_texts = defaultdict(lambda: defaultdict(int))
        for failure in self.failures:
            entity_texts[failure.entity_type][failure.entity_text] += 1

        common_failures = {}
        for entity_type, texts in entity_texts.items():
            sorted_texts = sorted(texts.items(), key=lambda x: -x[1])[:5]
            common_failures[entity_type] = [
                {"text": t, "count": c} for t, c in sorted_texts
            ]

        return {
            "total_failures": len(self.failures),
            "by_entity_type": dict(by_type),
            "by_failure_type": dict(by_failure_type),
            "by_context": dict(by_context),
            "by_type_and_failure": {k: dict(v) for k, v in by_type_and_failure.items()},
            "common_failures": common_failures,
        }

    def apply_decay(self) -> None:
        """Apply decay to failure counts.

        Called after each evaluation round to gradually reduce
        the influence of old failures.
        """
        for entity_type in self.failure_counts:
            for failure_type in self.failure_counts[entity_type]:
                self.failure_counts[entity_type][failure_type] = int(
                    self.failure_counts[entity_type][failure_type] * self.decay_factor
                )

    def reset(self) -> None:
        """Reset all failure tracking."""
        self.failures = []
        self.failure_counts = defaultdict(lambda: defaultdict(int))
        self.context_patterns = defaultdict(list)

    def configure_synpii(self, synpii_config: Dict) -> Dict:
        """Configure SynPII based on failure weights.

        Args:
            synpii_config: Base SynPII configuration

        Returns:
            Modified configuration with failure-weighted entity distribution
        """
        weights = self.get_sampling_weights()

        if not weights:
            return synpii_config

        # Adjust entity type probabilities
        if "entity_probabilities" not in synpii_config:
            synpii_config["entity_probabilities"] = {}

        for entity_type, weight in weights.items():
            synpii_config["entity_probabilities"][entity_type] = weight

        return synpii_config
