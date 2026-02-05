import pytest
from curriculum.difficulty import DifficultyDimensions, DifficultyLevel
from curriculum.controller import CurriculumController
from curriculum.sampler import FailureWeightedSampler

class TestCurriculumModule:
    """Tests for Curriculum & Difficulty Scheduling System."""

    def test_difficulty_levels(self):
        trivial = DifficultyDimensions.from_level(DifficultyLevel.TRIVIAL)
        expert = DifficultyDimensions.from_level(DifficultyLevel.EXPERT)
        assert trivial.overall_difficulty() < expert.overall_difficulty()

    def test_curriculum_controller(self):
        controller = CurriculumController(
            start_difficulty=DifficultyLevel.EASY,
            promotion_threshold=0.85,
            min_evals_for_promotion=2,
        )
        assert controller.current_level == DifficultyLevel.EASY
        
        # Test promotion logic
        controller.record_evaluation(f1=0.90, precision=0.92, recall=0.88)
        controller.record_evaluation(f1=0.91, precision=0.93, recall=0.89)
        assert controller.should_promote()
        controller.promote()
        assert controller.current_level == DifficultyLevel.MEDIUM

    def test_failure_sampler(self):
        sampler = FailureWeightedSampler()
        sampler.record_failure("OCCUPATION", "social_history", "missed", "Ingenieur")
        weights = sampler.get_sampling_weights()
        assert "OCCUPATION" in weights
