"""Difficulty Dimensions for Curriculum Learning.

Defines multi-dimensional difficulty control for synthetic data generation,
enabling structured progression from easy to hard cases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class DifficultyLevel(str, Enum):
    """Discrete difficulty levels for curriculum learning."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class DifficultyDimensions:
    """Multi-dimensional difficulty specification.

    Each dimension controls a specific aspect of difficulty,
    allowing fine-grained control over synthetic data generation.

    All dimensions are on a 1-5 scale:
    - 1: Trivial (easiest)
    - 3: Medium (default)
    - 5: Very Hard

    Example:
        # Easy configuration
        easy = DifficultyDimensions(
            entity_count=2,
            entity_variety=2,
            format_variation=1,
        )

        # Hard configuration
        hard = DifficultyDimensions(
            entity_count=5,
            entity_variety=5,
            format_variation=4,
            perturbation_level=4,
            overlap_frequency=4,
        )
    """
    # Entity complexity
    entity_count: int = 3           # 1-5: Entities per document
    entity_variety: int = 3         # 1-5: Different entity types per document
    entity_density: int = 3         # 1-5: How close entities are to each other

    # Format complexity
    format_variation: int = 2       # 1-5: Non-standard formats
    case_variation: int = 2         # 1-5: Case variations (UPPER, lower, Mixed)
    abbreviation_level: int = 2     # 1-5: Use of abbreviations

    # Perturbations
    perturbation_level: int = 1     # 1-5: OCR/BPE/typo perturbations
    noise_level: int = 1            # 1-5: Random noise/filler text

    # Structural complexity
    overlap_frequency: int = 2      # 1-5: Overlapping entities
    context_ambiguity: int = 2      # 1-5: Ambiguous contexts
    nesting_depth: int = 1          # 1-5: Nested entities

    # Document complexity
    document_length: int = 3        # 1-5: Document length (1=short, 5=very long)
    section_count: int = 3          # 1-5: Number of document sections

    def __post_init__(self):
        """Validate dimension values are in range."""
        for name, value in self.__dict__.items():
            if not 1 <= value <= 5:
                raise ValueError(f"{name} must be between 1 and 5, got {value}")

    @classmethod
    def from_level(cls, level: DifficultyLevel) -> "DifficultyDimensions":
        """Create difficulty dimensions from a discrete level.

        Args:
            level: The difficulty level

        Returns:
            Configured DifficultyDimensions
        """
        presets = {
            DifficultyLevel.TRIVIAL: cls(
                entity_count=1,
                entity_variety=1,
                entity_density=1,
                format_variation=1,
                case_variation=1,
                abbreviation_level=1,
                perturbation_level=1,
                noise_level=1,
                overlap_frequency=1,
                context_ambiguity=1,
                nesting_depth=1,
                document_length=1,
                section_count=1,
            ),
            DifficultyLevel.EASY: cls(
                entity_count=2,
                entity_variety=2,
                entity_density=2,
                format_variation=1,
                case_variation=1,
                abbreviation_level=1,
                perturbation_level=1,
                noise_level=1,
                overlap_frequency=1,
                context_ambiguity=1,
                nesting_depth=1,
                document_length=2,
                section_count=2,
            ),
            DifficultyLevel.MEDIUM: cls(
                entity_count=3,
                entity_variety=3,
                entity_density=3,
                format_variation=2,
                case_variation=2,
                abbreviation_level=2,
                perturbation_level=2,
                noise_level=2,
                overlap_frequency=2,
                context_ambiguity=2,
                nesting_depth=1,
                document_length=3,
                section_count=3,
            ),
            DifficultyLevel.HARD: cls(
                entity_count=4,
                entity_variety=4,
                entity_density=4,
                format_variation=3,
                case_variation=3,
                abbreviation_level=3,
                perturbation_level=3,
                noise_level=3,
                overlap_frequency=3,
                context_ambiguity=3,
                nesting_depth=2,
                document_length=4,
                section_count=4,
            ),
            DifficultyLevel.EXPERT: cls(
                entity_count=5,
                entity_variety=5,
                entity_density=5,
                format_variation=5,
                case_variation=4,
                abbreviation_level=4,
                perturbation_level=5,
                noise_level=4,
                overlap_frequency=5,
                context_ambiguity=5,
                nesting_depth=3,
                document_length=5,
                section_count=5,
            ),
        }
        return presets[level]

    def overall_difficulty(self) -> float:
        """Calculate overall difficulty score (0-1)."""
        total = sum(self.__dict__.values())
        max_total = 5 * len(self.__dict__)
        min_total = 1 * len(self.__dict__)
        return (total - min_total) / (max_total - min_total)

    def to_synpii_config(self) -> Dict:
        """Convert to SynPII generator configuration.

        Returns:
            Dict suitable for SynPII configuration
        """
        return {
            # Entity configuration
            "min_entities": max(1, self.entity_count - 1),
            "max_entities": self.entity_count + 2,
            "entity_variety": self.entity_variety,

            # Format configuration
            "use_standard_formats": self.format_variation < 3,
            "include_variations": self.format_variation >= 2,
            "case_variation": self.case_variation >= 3,

            # Perturbation configuration
            "perturbation_probability": 0.1 * self.perturbation_level,
            "ocr_errors": self.perturbation_level >= 3,
            "typo_errors": self.perturbation_level >= 2,

            # Document configuration
            "document_length": ["short", "medium", "medium", "long", "very_long"][
                self.document_length - 1
            ],
            "sections": self.section_count,

            # Overlap configuration
            "allow_overlapping": self.overlap_frequency >= 2,
            "overlap_probability": 0.15 * self.overlap_frequency,
        }

    def adjust_dimension(
        self,
        dimension: str,
        delta: int = 1,
    ) -> "DifficultyDimensions":
        """Create a new DifficultyDimensions with one dimension adjusted.

        Args:
            dimension: Name of dimension to adjust
            delta: Amount to adjust (+1 or -1)

        Returns:
            New DifficultyDimensions with adjusted value
        """
        new_values = dict(self.__dict__)
        if dimension in new_values:
            new_values[dimension] = max(1, min(5, new_values[dimension] + delta))
        return DifficultyDimensions(**new_values)

    def increase_all(self, delta: int = 1) -> "DifficultyDimensions":
        """Increase all dimensions by delta.

        Args:
            delta: Amount to increase each dimension

        Returns:
            New DifficultyDimensions with all values increased
        """
        new_values = {k: max(1, min(5, v + delta)) for k, v in self.__dict__.items()}
        return DifficultyDimensions(**new_values)

    def decrease_all(self, delta: int = 1) -> "DifficultyDimensions":
        """Decrease all dimensions by delta.

        Args:
            delta: Amount to decrease each dimension

        Returns:
            New DifficultyDimensions with all values decreased
        """
        return self.increase_all(-delta)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict) -> "DifficultyDimensions":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
