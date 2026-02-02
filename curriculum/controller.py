"""Curriculum Controller.

Controls difficulty progression through curriculum learning,
automatically adjusting difficulty based on performance.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from curriculum.difficulty import DifficultyDimensions, DifficultyLevel
from curriculum.sampler import FailureWeightedSampler


@dataclass
class EvaluationRecord:
    """Record of an evaluation at a difficulty level."""
    level: DifficultyLevel
    dimensions: DifficultyDimensions
    f1_score: float
    precision: float
    recall: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class CurriculumController:
    """Controller for curriculum-based learning progression.

    Manages difficulty progression through:
    1. Starting at a configured difficulty level
    2. Advancing when performance exceeds threshold
    3. Demoting when performance drops
    4. Integrating failure-weighted sampling

    Example:
        controller = CurriculumController(
            start_difficulty=DifficultyLevel.EASY,
            promotion_threshold=0.85,
            demotion_threshold=0.60,
        )

        # Get current difficulty
        difficulty = controller.get_current_difficulty()

        # Record evaluation results
        controller.record_evaluation(f1=0.82, precision=0.85, recall=0.79)

        # Check if level changed
        if controller.should_promote():
            controller.promote()
            print("Advanced to harder difficulty!")
    """

    def __init__(
        self,
        start_difficulty: DifficultyLevel = DifficultyLevel.EASY,
        promotion_threshold: float = 0.85,  # F1 to advance
        demotion_threshold: float = 0.60,   # F1 to go back
        learning_velocity_window: int = 5,  # Samples for velocity calc
        min_evals_for_promotion: int = 3,   # Minimum evals before promoting
        failure_sampler: Optional[FailureWeightedSampler] = None,
    ):
        """Initialize the curriculum controller.

        Args:
            start_difficulty: Initial difficulty level
            promotion_threshold: F1 score required to advance
            demotion_threshold: F1 score below which to demote
            learning_velocity_window: Number of evaluations for velocity
            min_evals_for_promotion: Minimum evaluations before promotion
            failure_sampler: Optional failure-weighted sampler
        """
        self.current_level = start_difficulty
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.learning_velocity_window = learning_velocity_window
        self.min_evals_for_promotion = min_evals_for_promotion

        self.failure_sampler = failure_sampler or FailureWeightedSampler()

        # Track history
        self.level_history: Dict[DifficultyLevel, List[EvaluationRecord]] = defaultdict(list)
        self.all_evaluations: List[EvaluationRecord] = []

        # Current difficulty dimensions (can be customized)
        self._current_dimensions: Optional[DifficultyDimensions] = None

    def get_current_difficulty(self) -> DifficultyDimensions:
        """Get the current difficulty configuration.

        Returns:
            DifficultyDimensions for current level, adjusted by failure bias
        """
        if self._current_dimensions:
            return self._current_dimensions

        base = DifficultyDimensions.from_level(self.current_level)
        return self._apply_failure_bias(base)

    def _apply_failure_bias(
        self,
        difficulty: DifficultyDimensions,
    ) -> DifficultyDimensions:
        """Apply failure-based adjustments to difficulty.

        If certain entity types are failing more, increase related
        difficulty dimensions.
        """
        analysis = self.failure_sampler.get_failure_analysis()

        if not analysis["by_entity_type"]:
            return difficulty

        # Find most problematic failure types
        failure_types = analysis.get("by_failure_type", {})

        # Adjust dimensions based on failure patterns
        if failure_types.get("wrong_boundary", 0) > 5:
            difficulty = difficulty.adjust_dimension("entity_density", 1)

        if failure_types.get("wrong_type", 0) > 5:
            difficulty = difficulty.adjust_dimension("context_ambiguity", 1)

        if failure_types.get("missed", 0) > 10:
            difficulty = difficulty.adjust_dimension("entity_variety", 1)

        return difficulty

    def record_evaluation(
        self,
        f1: float,
        precision: float,
        recall: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record an evaluation result.

        Args:
            f1: F1 score
            precision: Precision score
            recall: Recall score
            metadata: Additional metadata
        """
        record = EvaluationRecord(
            level=self.current_level,
            dimensions=self.get_current_difficulty(),
            f1_score=f1,
            precision=precision,
            recall=recall,
            metadata=metadata or {},
        )

        self.level_history[self.current_level].append(record)
        self.all_evaluations.append(record)

    def should_promote(self) -> bool:
        """Check if should advance to harder difficulty.

        Returns:
            True if should promote
        """
        if self.current_level == DifficultyLevel.EXPERT:
            return False  # Already at max

        history = self.level_history[self.current_level]

        if len(history) < self.min_evals_for_promotion:
            return False

        # Check if recent performance is above threshold
        recent = history[-self.min_evals_for_promotion:]
        avg_f1 = sum(r.f1_score for r in recent) / len(recent)

        return avg_f1 >= self.promotion_threshold

    def should_demote(self) -> bool:
        """Check if should fall back to easier difficulty.

        Returns:
            True if should demote
        """
        if self.current_level == DifficultyLevel.TRIVIAL:
            return False  # Already at min

        history = self.level_history[self.current_level]

        if len(history) < 2:
            return False

        # Check if recent performance is below threshold
        recent = history[-3:] if len(history) >= 3 else history
        avg_f1 = sum(r.f1_score for r in recent) / len(recent)

        return avg_f1 < self.demotion_threshold

    def promote(self) -> DifficultyLevel:
        """Advance to next difficulty level.

        Returns:
            New difficulty level
        """
        levels = list(DifficultyLevel)
        current_idx = levels.index(self.current_level)

        if current_idx < len(levels) - 1:
            self.current_level = levels[current_idx + 1]
            self._current_dimensions = None  # Reset custom dimensions

        return self.current_level

    def demote(self) -> DifficultyLevel:
        """Fall back to previous difficulty level.

        Returns:
            New difficulty level
        """
        levels = list(DifficultyLevel)
        current_idx = levels.index(self.current_level)

        if current_idx > 0:
            self.current_level = levels[current_idx - 1]
            self._current_dimensions = None  # Reset custom dimensions

        return self.current_level

    def auto_adjust(self) -> Tuple[DifficultyLevel, str]:
        """Automatically adjust difficulty based on performance.

        Returns:
            Tuple of (new level, action taken)
        """
        if self.should_promote():
            self.promote()
            return self.current_level, "promoted"
        elif self.should_demote():
            self.demote()
            return self.current_level, "demoted"
        return self.current_level, "maintained"

    def compute_learning_velocity(self) -> float:
        """Calculate how fast performance is improving.

        Returns:
            Learning velocity (positive = improving, negative = declining)
        """
        history = self.level_history[self.current_level]

        if len(history) < self.learning_velocity_window:
            return 0.0

        recent = history[-self.learning_velocity_window:]
        f1_scores = [r.f1_score for r in recent]

        # Simple linear regression slope
        n = len(f1_scores)
        x_mean = (n - 1) / 2
        y_mean = sum(f1_scores) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(f1_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_progress_summary(self) -> Dict:
        """Get a summary of curriculum progress.

        Returns:
            Progress summary dict
        """
        total_evals = len(self.all_evaluations)

        # Calculate per-level statistics
        level_stats = {}
        for level, history in self.level_history.items():
            if history:
                f1_scores = [r.f1_score for r in history]
                level_stats[level.value] = {
                    "evaluations": len(history),
                    "avg_f1": sum(f1_scores) / len(f1_scores),
                    "max_f1": max(f1_scores),
                    "min_f1": min(f1_scores),
                }

        return {
            "current_level": self.current_level.value,
            "total_evaluations": total_evals,
            "level_statistics": level_stats,
            "learning_velocity": self.compute_learning_velocity(),
            "ready_for_promotion": self.should_promote(),
            "at_risk_of_demotion": self.should_demote(),
        }

    def set_custom_dimensions(self, dimensions: DifficultyDimensions) -> None:
        """Set custom difficulty dimensions.

        Args:
            dimensions: Custom dimensions to use
        """
        self._current_dimensions = dimensions

    def reset_to_level(self, level: DifficultyLevel) -> None:
        """Reset to a specific difficulty level.

        Args:
            level: Level to reset to
        """
        self.current_level = level
        self._current_dimensions = None

    def get_synpii_config(self) -> Dict:
        """Get SynPII configuration for current difficulty.

        Returns:
            Configuration dict for SynPII
        """
        difficulty = self.get_current_difficulty()
        config = difficulty.to_synpii_config()

        # Apply failure weighting
        config = self.failure_sampler.configure_synpii(config)

        return config
