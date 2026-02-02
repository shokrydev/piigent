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
        from synpii.weakness import WeaknessTargetedGenerator, WeaknessType, WeaknessReport
        from synpii.core.lexicon import GermanLexicon
        from synpii.generators import GeneratorRegistry

        # Initialize SynPII components
        values_dir = Path(__file__).parent.parent / "synpii" / "values"
        self.lexicon = GermanLexicon(values_dir)
        self.registry = GeneratorRegistry(lexicon=self.lexicon)
        self.synpii = SynPII(preset="clinical_de", seed=seed)
        self.targeted_gen = WeaknessTargetedGenerator(
            generators=self.registry,
            lexicon=self.lexicon,
        )

        # SynPII weakness types
        self.WeaknessType = WeaknessType
        self.WeaknessReport = WeaknessReport

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

    def _map_weakness_type(self, weakness_type_str: str) -> Optional["WeaknessType"]:
        """Map string weakness type to SynPII WeaknessType enum."""
        mapping = {
            "overlap_conflict": self.WeaknessType.OVERLAP_CONFLICT,
            "format_variation": self.WeaknessType.FORMAT_VARIATION,
            "context_dependency": self.WeaknessType.CONTEXT_DEPENDENCY,
            "coverage_gap": self.WeaknessType.COVERAGE_GAP,
            "entity_confusion": self.WeaknessType.ENTITY_CONFUSION,
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
        synpii_weakness_type = self._map_weakness_type(weakness_type_str)

        if synpii_weakness_type is None:
            logger.warning(f"Unknown weakness type: {weakness_type_str}")
            return self._fallback_generation(entity_type, count)

        # Create SynPII WeaknessReport
        report = self.WeaknessReport(
            weakness_type=synpii_weakness_type,
            entity_type=entity_type,
            description=weakness.get("description", ""),
            evidence=weakness.get("evidence", {}),
        )

        # Generate using SynPII
        synpii_cases = self.targeted_gen.generate_for_weakness(report, count=count)

        # Convert to TestCase format
        test_cases = []
        for case in synpii_cases:
            # Extract entity values from text
            expected = {entity_type: [case.text]}

            # Check for overlapping entity types
            if synpii_weakness_type == self.WeaknessType.OVERLAP_CONFLICT:
                evidence = weakness.get("evidence", {})
                if isinstance(evidence, dict) and "conflicting_type" in evidence:
                    conflicting_type = evidence["conflicting_type"]
                    expected[conflicting_type] = []  # Expected to potentially interfere

            test_cases.append(TestCase(
                text=case.text,
                expected_entities=expected,
                description=case.description if hasattr(case, 'description') else f"SynPII-generated {synpii_weakness_type.value}",
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
