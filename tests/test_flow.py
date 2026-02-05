import pytest
from components.section_parser import SectionParser
from components.domain_rules import DomainRulesEngine
from components.overlap_resolver import OverlapResolver, ResolutionStrategy
from components.consistency import ConsistencyValidator

class TestFlowModule:
    """Tests for the Multi-Step Flow Components."""

    def test_section_parser(self):
        text = "Anamnese: Patient hat Schmerzen.\nBefund: Keine Auffälligkeiten."
        parser = SectionParser()
        doc = parser.parse(text)
        assert len(doc.sections) == 2
        assert doc.sections[0].header.startswith("Anamnese")

    def test_overlap_resolver(self):
        # Specific overlap conflict test
        entities = [
            {"text": "10115", "start": 0, "end": 5, "label": "DE_POSTAL_CODE", "score": 0.8},
            {"text": "10115", "start": 0, "end": 5, "label": "NUMBER", "score": 0.9}
        ]
        resolver = OverlapResolver(strategy=ResolutionStrategy.PREFER_SPECIFIC)
        resolved = resolver.resolve(entities)
        
        # Should prefer DE_POSTAL_CODE over generic NUMBER despite lower score if configured
        assert len(resolved) == 1
        assert resolved[0]["label"] == "DE_POSTAL_CODE"
