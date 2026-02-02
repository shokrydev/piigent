"""SynPII Curriculum Integration Wrapper.

Integrates SynPII with the curriculum system for:
- Difficulty-based document generation
- Failure-weighted entity sampling
- Distribution matching
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from curriculum.controller import CurriculumController
from curriculum.difficulty import DifficultyDimensions, DifficultyLevel

logger = logging.getLogger(__name__)


@dataclass
class GeneratedDocument:
    """A generated document with annotations."""
    text: str
    entities: List[Dict]  # List of {text, entity_type, start, end}
    metadata: Dict


class CurriculumSynPIIWrapper:
    """Wrapper for SynPII with curriculum-based generation.

    Provides curriculum-aware document generation that:
    1. Adjusts complexity based on current difficulty level
    2. Oversamples failure-prone entity types
    3. Tracks distribution statistics

    Example:
        controller = CurriculumController()
        wrapper = CurriculumSynPIIWrapper(controller)

        # Generate documents at current difficulty
        docs = wrapper.generate_batch(10)

        # Generate targeted test cases
        cases = wrapper.generate_for_entity_type("OCCUPATION", count=5)
    """

    def __init__(
        self,
        curriculum_controller: CurriculumController,
        base_config: Optional[Dict] = None,
    ):
        """Initialize the curriculum SynPII wrapper.

        Args:
            curriculum_controller: The curriculum controller
            base_config: Base configuration for SynPII
        """
        self.controller = curriculum_controller
        self.base_config = base_config or {}

        # Import SynPII lazily to avoid circular imports
        self._synpii = None
        self._synpii_class = None

    def _get_synpii(self):
        """Lazily load SynPII."""
        if self._synpii is None:
            try:
                from synpii import SynPII
                self._synpii_class = SynPII
                self._synpii = SynPII(preset="clinical_de")
            except ImportError:
                logger.warning("SynPII not available")
                return None
        return self._synpii

    def generate_document(
        self,
        difficulty: Optional[DifficultyDimensions] = None,
    ) -> Optional[GeneratedDocument]:
        """Generate a single document.

        Args:
            difficulty: Optional difficulty override

        Returns:
            Generated document or None if SynPII unavailable
        """
        synpii = self._get_synpii()
        if synpii is None:
            return None

        # Get configuration
        if difficulty is None:
            difficulty = self.controller.get_current_difficulty()

        config = difficulty.to_synpii_config()
        config = self.controller.failure_sampler.configure_synpii(config)
        config.update(self.base_config)

        # Generate document
        try:
            doc = synpii.generate(config=config)
            return GeneratedDocument(
                text=doc.text if hasattr(doc, 'text') else str(doc),
                entities=doc.entities if hasattr(doc, 'entities') else [],
                metadata={
                    "difficulty_level": self.controller.current_level.value,
                    "overall_difficulty": difficulty.overall_difficulty(),
                    "config": config,
                },
            )
        except Exception as e:
            logger.error(f"SynPII generation failed: {e}")
            return None

    def generate_batch(
        self,
        count: int,
        difficulty: Optional[DifficultyDimensions] = None,
    ) -> List[GeneratedDocument]:
        """Generate multiple documents.

        Args:
            count: Number of documents to generate
            difficulty: Optional difficulty override

        Returns:
            List of generated documents
        """
        documents = []
        for _ in range(count):
            doc = self.generate_document(difficulty)
            if doc:
                documents.append(doc)
        return documents

    def generate_for_entity_type(
        self,
        entity_type: str,
        count: int = 5,
        context: Optional[str] = None,
    ) -> List[GeneratedDocument]:
        """Generate documents targeting a specific entity type.

        Args:
            entity_type: Entity type to target
            count: Number of documents
            context: Optional context (e.g., "social_history")

        Returns:
            List of generated documents
        """
        synpii = self._get_synpii()
        if synpii is None:
            return []

        # Configure for specific entity type
        difficulty = self.controller.get_current_difficulty()
        config = difficulty.to_synpii_config()

        # Boost target entity type
        if "entity_probabilities" not in config:
            config["entity_probabilities"] = {}
        config["entity_probabilities"][entity_type] = 3.0  # 3x weight

        # Set context if provided
        if context:
            config["target_section"] = context

        documents = []
        for _ in range(count):
            try:
                doc = synpii.generate(config=config)
                documents.append(GeneratedDocument(
                    text=doc.text if hasattr(doc, 'text') else str(doc),
                    entities=doc.entities if hasattr(doc, 'entities') else [],
                    metadata={
                        "target_entity_type": entity_type,
                        "context": context,
                    },
                ))
            except Exception as e:
                logger.error(f"Targeted generation failed: {e}")

        return documents

    def generate_overlap_tests(
        self,
        entity_type_a: str,
        entity_type_b: str,
        count: int = 5,
    ) -> List[GeneratedDocument]:
        """Generate documents with overlapping entities.

        Args:
            entity_type_a: First entity type
            entity_type_b: Second entity type
            count: Number of documents

        Returns:
            List of documents with overlap scenarios
        """
        synpii = self._get_synpii()
        if synpii is None:
            return []

        config = {
            "allow_overlapping": True,
            "overlap_probability": 0.8,
            "entity_probabilities": {
                entity_type_a: 2.0,
                entity_type_b: 2.0,
            },
        }

        documents = []
        for _ in range(count):
            try:
                doc = synpii.generate(config=config)
                documents.append(GeneratedDocument(
                    text=doc.text if hasattr(doc, 'text') else str(doc),
                    entities=doc.entities if hasattr(doc, 'entities') else [],
                    metadata={
                        "test_type": "overlap",
                        "entity_types": [entity_type_a, entity_type_b],
                    },
                ))
            except Exception as e:
                logger.error(f"Overlap test generation failed: {e}")

        return documents

    def generate_format_variation_tests(
        self,
        entity_type: str,
        count: int = 5,
    ) -> List[GeneratedDocument]:
        """Generate documents with format variations.

        Args:
            entity_type: Entity type to test
            count: Number of documents

        Returns:
            List of documents with format variations
        """
        synpii = self._get_synpii()
        if synpii is None:
            return []

        config = {
            "use_standard_formats": False,
            "include_variations": True,
            "case_variation": True,
            "entity_probabilities": {entity_type: 3.0},
        }

        documents = []
        for _ in range(count):
            try:
                doc = synpii.generate(config=config)
                documents.append(GeneratedDocument(
                    text=doc.text if hasattr(doc, 'text') else str(doc),
                    entities=doc.entities if hasattr(doc, 'entities') else [],
                    metadata={
                        "test_type": "format_variation",
                        "entity_type": entity_type,
                    },
                ))
            except Exception as e:
                logger.error(f"Format variation generation failed: {e}")

        return documents

    def generate_perturbation_tests(
        self,
        perturbation_type: str = "ocr",
        count: int = 5,
    ) -> List[GeneratedDocument]:
        """Generate documents with perturbations.

        Args:
            perturbation_type: Type of perturbation ("ocr", "typo", "bpe")
            count: Number of documents

        Returns:
            List of perturbed documents
        """
        synpii = self._get_synpii()
        if synpii is None:
            return []

        config = {
            "perturbation_probability": 0.5,
            "ocr_errors": perturbation_type == "ocr",
            "typo_errors": perturbation_type == "typo",
            "bpe_errors": perturbation_type == "bpe",
        }

        documents = []
        for _ in range(count):
            try:
                doc = synpii.generate(config=config)
                documents.append(GeneratedDocument(
                    text=doc.text if hasattr(doc, 'text') else str(doc),
                    entities=doc.entities if hasattr(doc, 'entities') else [],
                    metadata={
                        "test_type": "perturbation",
                        "perturbation_type": perturbation_type,
                    },
                ))
            except Exception as e:
                logger.error(f"Perturbation test generation failed: {e}")

        return documents

    def get_generation_statistics(self) -> Dict:
        """Get statistics about document generation.

        Returns:
            Statistics dict
        """
        return {
            "current_level": self.controller.current_level.value,
            "difficulty_config": self.controller.get_current_difficulty().to_dict(),
            "failure_weights": self.controller.failure_sampler.get_sampling_weights(),
            "failure_analysis": self.controller.failure_sampler.get_failure_analysis(),
        }
