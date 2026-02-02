"""Curriculum & Difficulty Scheduling System.

Provides structured progression and failure-weighted sampling:
- Multi-dimensional difficulty control
- Curriculum-based learning progression
- Failure-weighted oversampling of hard cases
"""

from curriculum.difficulty import DifficultyDimensions, DifficultyLevel
from curriculum.controller import CurriculumController
from curriculum.sampler import FailureWeightedSampler

__all__ = [
    "DifficultyDimensions",
    "DifficultyLevel",
    "CurriculumController",
    "FailureWeightedSampler",
]
