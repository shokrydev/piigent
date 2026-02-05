import pytest
from agents.critics.rationale import RationaleAgent
from agents.critics.error_taxonomy import ErrorTaxonomyAgent
from agents.critics.fix_proposal import FixProposalAgent
from agents.critics.verification import VerificationAgent

class TestAgentsModule:
    """Tests for Self-Critique Agents."""

    def test_rationale_agent(self):
        agent = RationaleAgent()
        # Mocking the actual LLM call for unit testing
        explanation = agent.explain_prediction(text="Test", entity={"text": "Foo", "label": "BAR", "start": 0, "end": 3})
        assert explanation is not None

    def test_taxonomy_agent(self):
        agent = ErrorTaxonomyAgent()
        category = agent.classify_error(expected={"text": "A", "start": 0, "end": 1}, detected=None, context="Context A")
        assert category.error_type.name == "FALSE_NEGATIVE"
