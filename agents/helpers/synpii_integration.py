"""SynPII Integration for Weakness Analyzer.

Provides SynPII-powered test case generation for the Weakness Analyzer agent.
Uses grammar-aware generation with proper checksums and realistic variations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A targeted test case for weakness exploration."""
    text: str
    expected_entities: Dict[str, List[str]]
    description: str


class SynPIITestGenerator:
    """SynPII-powered test case generator for weakness exploration.

    Uses SynPII's weakness-targeted generation to create realistic
    test cases that expose specific detection weaknesses.
    """

    def __init__(self, seed: int = None):
        """Initialize the SynPII test generator.

        Args:
            seed: Random seed for reproducibility.
        """
        from synpii import SynPII
        from synpii.adversarial import AdversarialType, AdversarialScenario

        # Use the SynPII facade for unified component management
        # It handles paths, seeds, and lazy-loading of sub-components
        self.synpii = SynPII(preset="clinical_de", seed=seed)
        
        # Access shared components from the facade
        self.lexicon = self.synpii.lexicon
        self.registry = self.synpii.generators
        self.adversarial_gen = self.synpii.adversarial_generator

        # SynPII research types
        self.AdversarialType = AdversarialType
        self.AdversarialScenario = AdversarialScenario

    def generate_document(self) -> Dict:
        """Generate a synthetic clinical document with annotations.

        Returns:
            Dict with 'text', 'annotations', and metadata.
        """
        doc = self.synpii.generate_document()
        return {
            "text": doc.text,
            "annotations": [
                {
                    "entity_type": ann.entity_type,
                    "text": ann.text,
                    "start": ann.start,
                    "end": ann.end,
                }
                for ann in doc.annotations
            ],
            "template_type": doc.template_type,
            "id": doc.id,
        }

    def generate_documents(self, count: int) -> List[Dict]:
        """Generate multiple synthetic documents.

        Args:
            count: Number of documents to generate.

        Returns:
            List of document dicts.
        """
        return [self.generate_document() for _ in range(count)]

    def _map_adversarial_type(self, weakness_type_str: str) -> Optional["AdversarialType"]:
        """Map PIIgent weakness type to SynPII AdversarialType enum."""
        mapping = {
            "overlap_conflict": self.AdversarialType.OVERLAP_CONFLICT,
            "format_variation": self.AdversarialType.FORMAT_VARIATION,
            "context_dependency": self.AdversarialType.CONTEXT_DEPENDENCY,
            "coverage_gap": self.AdversarialType.COVERAGE_GAP,
            "entity_confusion": self.AdversarialType.ENTITY_CONFUSION,
        }
        return mapping.get(weakness_type_str)

    def generate_for_weakness(self, weakness: Dict, count: int = 10) -> List[TestCase]:
        """Generate test cases targeting a specific weakness.

        Args:
            weakness: Dict with 'weakness_type', 'entity_type', etc.
            count: Number of test cases to generate.

        Returns:
            List of TestCase objects.
        """
        weakness_type_str = weakness.get("weakness_type", "")
        entity_type = weakness.get("entity_type", "")
        adv_type = self._map_adversarial_type(weakness_type_str)

        if adv_type is None:
            logger.warning(f"Unknown weakness type: {weakness_type_str}")
            return self._fallback_generation(entity_type, count)

        # Create SynPII AdversarialScenario
        scenario = self.AdversarialScenario(
            adversarial_type=adv_type,
            entity_type=entity_type,
            description=weakness.get("description", ""),
            evidence=weakness.get("evidence", {}),
        )

        # Generate using SynPII adversarial generator
        synpii_samples = self.adversarial_gen.generate_for_scenario(scenario, count=count)

        # Convert to TestCase format
        test_cases = []
        for sample in synpii_samples:
            # Extract entity values from text
            expected = {entity_type: [sample.text]}

            # Check for overlapping entity types
            if adv_type == self.AdversarialType.OVERLAP_CONFLICT:
                evidence = weakness.get("evidence", {})
                if isinstance(evidence, dict) and "conflicting_type" in evidence:
                    conflicting_type = evidence["conflicting_type"]
                    expected[conflicting_type] = []  # Expected to potentially interfere

            test_cases.append(TestCase(
                text=sample.text,
                expected_entities=expected,
                description=sample.metadata.get("pattern", f"Adversarial {adv_type.value}"),
            ))

        return test_cases

    def _fallback_generation(self, entity_type: str, count: int) -> List[TestCase]:
        """Fallback generation when SynPII doesn't support the weakness type."""
        test_cases = []

        for _ in range(count):
            if self.registry.is_available(entity_type):
                entity = self.registry.generate(entity_type)
                test_cases.append(TestCase(
                    text=f"Test: {entity.value}",
                    expected_entities={entity_type: [entity.value]},
                    description=f"Isolated {entity_type} test",
                ))

        return test_cases

    def generate_overlap_tests(
        self,
        entity_type: str,
        overlapping_type: str,
        count: int = 10,
    ) -> List[TestCase]:
        """Generate tests for entity overlap scenarios.

        Args:
            entity_type: Primary entity type (e.g., "DE_POSTAL_CODE").
            overlapping_type: Entity type that may overlap (e.g., "LOCATION").
            count: Number of test cases.

        Returns:
            List of TestCase objects.
        """
        weakness = {
            "weakness_type": "overlap_conflict",
            "entity_type": entity_type,
            "description": f"{entity_type} overlaps with {overlapping_type}",
            "evidence": {"conflicting_type": overlapping_type},
        }
        return self.generate_for_weakness(weakness, count)

    def generate_format_tests(self, entity_type: str, count: int = 10) -> List[TestCase]:
        """Generate tests for format variations.

        Args:
            entity_type: Entity type to test formats for.
            count: Number of test cases.

        Returns:
            List of TestCase objects.
        """
        weakness = {
            "weakness_type": "format_variation",
            "entity_type": entity_type,
            "description": f"{entity_type} format variations",
        }
        return self.generate_for_weakness(weakness, count)

    def generate_context_tests(self, entity_type: str, count: int = 10) -> List[TestCase]:
        """Generate tests for context dependency.

        Args:
            entity_type: Entity type to test.
            count: Number of test cases.

        Returns:
            List of TestCase objects.
        """
        weakness = {
            "weakness_type": "context_dependency",
            "entity_type": entity_type,
            "description": f"{entity_type} context variations",
        }
        return self.generate_for_weakness(weakness, count)


def get_synpii_generator(seed: int = None) -> SynPIITestGenerator:
    """Factory function to create a SynPII test generator.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Configured SynPIITestGenerator instance.
    """
    return SynPIITestGenerator(seed=seed)
