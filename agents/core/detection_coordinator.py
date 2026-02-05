"""Detection Coordinator - Orchestrates PII detection using AnonerWrapper.

Acts as a LangGraph agent node that delegates heavy-lifting to the AnonerWrapper.
"""

import logging
from typing import Literal

from graph.state import FlowState
from agents.helpers.aggregator import EntityAggregator
from components.reflective_resolution import ReflectiveResolver
from wrappers.anoner_wrapper import AnonerWrapper

logger = logging.getLogger(__name__)


class DetectionCoordinator:
    """Coordinates parallel PII detection using the AnonerWrapper."""

    def __init__(
        self,
        preset: str = "clinical",
        use_ministral: bool = False,
        ministral_model: str = "ministral-3:8b",
        ollama_url: str = "http://localhost:11434",
        use_gliner: bool = False,
    ):
        self.preset = preset
        self.use_ministral = use_ministral
        self.use_gliner = use_gliner
        
        # Initialize Wrapper
        self.analyzer = AnonerWrapper(
            preset=preset,
            ollama_url=ollama_url,
            model=ministral_model
        )
        
        # Initialize Agentic Resolver
        resolver = None
        if use_ministral:
             resolver = ReflectiveResolver(
                 ollama_url=ollama_url,
                 model=ministral_model
             )
        
        self.aggregator = EntityAggregator(resolver=resolver)

        logger.info(f"DetectionCoordinator initialized (preset={preset})")

    def __call__(self, state: FlowState) -> dict:
        """LangGraph agent node - execute detection flow."""
        document = state["document"]
        threshold = state.get("confidence_threshold", 0.7)

        # Handle hints from Quality Audit (Recursive loop)
        hints = state.get("missed_entities", [])
        if hints:
            logger.info(f"Retrying detection with {len(hints)} hints from audit")

        logger.info(f"Detecting PII in {len(document)} chars")

        # Delegate to Wrapper
        results = self.analyzer.analyze_all(
            text=document,
            use_llm=self.use_ministral,
            use_ner=self.use_gliner
        )

        # Aggregate
        entities = self.aggregator.aggregate(results, text=document)
        
        # Incorporate hints if present
        if hints:
            # Simple addition for now: ensure hints are in the final list
            for hint in hints:
                if not any(e.start == hint.start and e.end == hint.end for e in entities):
                    entities.append(hint)

        min_conf = min((e.score for e in entities), default=1.0)
        needs_validation = min_conf < threshold

        message = (
            f"Found {len(entities)} entities. "
            f"Min confidence: {min_conf:.2f}."
        )
        logger.info(message)

        return {
            "detected_entities": entities,
            "min_confidence": min_conf,
            "needs_human_validation": needs_validation,
            "messages": [{"role": "assistant", "content": message}],
            # Clear hints after consumption
            "missed_entities": [],
        }


def create_detection_coordinator(
    preset: str = "clinical",
    use_ministral: bool = False,
    ministral_model: str = "ministral-3:8b",
    ollama_url: str = "http://localhost:11434",
    use_gliner: bool = False,
    **kwargs,
) -> DetectionCoordinator:
    """Factory function for DetectionCoordinator."""
    return DetectionCoordinator(
        preset=preset,
        use_ministral=use_ministral,
        ministral_model=ministral_model,
        ollama_url=ollama_url,
        use_gliner=use_gliner,
    )


def route_after_detection(state: FlowState) -> Literal["human_validation", "anonymize"]:
    """Routing function for LangGraph conditional edge."""
    if state.get("needs_human_validation", False):
        return "human_validation"
    return "anonymize"
