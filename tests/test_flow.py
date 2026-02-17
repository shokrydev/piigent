import pytest
from components.section_parser import SectionParser
from components.domain_rules import DomainRulesEngine
from components.reflective_resolution import OverlapResolver, ResolutionStrategy
from components.consistency import ConsistencyValidator
from unittest.mock import MagicMock
from wrappers.anoner_wrapper import AnonerWrapper

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
    def test_age_detection_propagation(self, monkeypatch):
        """Verify AGE entity is passed to the LLM recognizer."""
        
        # Mock the availability constant
        monkeypatch.setattr('wrappers.anoner_wrapper.PRESIDIO_AVAILABLE', True)
        
        # Create and set the mock recognizer
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = []
        
        # We need to mock the class to return our mock instance
        # Note: We use a lambda or a simple mock that returns mock_instance when called
        monkeypatch.setattr(
            'presidio_analyzer.predefined_recognizers.OllamaNERecognizer',
            lambda *args, **kwargs: mock_instance
        )
        
        text = "Patient ist 45 Jahre alt."
        wrapper = AnonerWrapper(model="ministral-test", ollama_url="http://localhost:11434")
        wrapper.analyze_llm(text)
        
        # Verify analyze was called with AGE in entities
        args, kwargs = mock_instance.analyze.call_args
        entities = kwargs.get('entities')
        assert "AGE" in entities
