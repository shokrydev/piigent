import pytest
from safeguards.overfitting import OverfittingDetector
from safeguards.leakage import LeakageChecker
from wrappers.llm_wrapper import PromptManagedLLM
from wrappers.synpii_wrapper import CurriculumSynPIIWrapper
from ui.hitl_interface import HITLTrigger

class TestSafeguardsModule:
    """Tests for System Safeguards."""
    
    def test_overfitting_detector(self):
        detector = OverfittingDetector()
        # Mocking check for simple logic test or using check_overfitting with dummies
        report = detector.check_overfitting(
            pipeline=lambda x: [],
            synthetic_samples=[{"text": "A", "entities": []}],
            real_samples=[{"text": "B", "entities": []}]
        )
        assert report is not None

class TestWrappersModule:
    """Tests for Integration Wrappers."""
    
    def test_llm_wrapper_initialization(self):
        import tempfile
        from prompts.store import PromptGenomeStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = PromptGenomeStore(f.name)
            llm = PromptManagedLLM(genome_store=store, model="ministral")
            assert llm.model == "ministral"

class TestUIModule:
    """Tests for UI Components."""
    
    def test_hitl_trigger(self):
        assert HITLTrigger.LOW_CONFIDENCE is not None
