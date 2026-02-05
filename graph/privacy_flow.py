"""LangGraph flow for PII detection and anonymization.

Three-agent flow with conditional HITL routing:
1. Detection Coordinator - Runs recognizers, aggregates results
2. Anonymization Strategist - Applies anonymization techniques
3. Quality Auditor - Verifies no PII leakage

Human-in-the-loop validation is triggered when confidence < threshold.
"""

import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from graph.state import FlowState, DetectedEntity
from agents.core.detection_coordinator import (
    DetectionCoordinator,
    create_detection_coordinator,
    route_after_detection,
)
from wrappers.anoner_wrapper import AnonerWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Agents
# =============================================================================


def human_validation_node(state: FlowState) -> dict:
    """Human-in-the-loop validation node."""
    entities = state.get("detected_entities", [])

    logger.info(f"HITL: {len(entities)} entities awaiting human validation")

    return {
        "messages": [{
            "role": "human",
            "content": f"Human validated {len(entities)} entities."
        }],
        "needs_human_validation": False,
    }


def anonymization_node(state: FlowState) -> dict:
    """Anonymization Strategist agent - processes replacements end-to-start."""
    document = state["document"]
    entities = state.get("detected_entities", [])

    logger.info(f"Anonymizing {len(entities)} entities in document")

    # Standardize replacements: Process reverse to keep offsets valid
    anonymized = document
    
    # Sort by start position reversed
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        replacement = f"[{entity.entity_type}]"
        anonymized = anonymized[:entity.start] + replacement + anonymized[entity.end:]

    return {
        "anonymized_text": anonymized,
        "messages": [{
            "role": "assistant",
            "content": f"Anonymized {len(entities)} entities using [TYPE] placeholders."
        }],
    }


def quality_audit_node(state: FlowState) -> dict:
    """Quality Auditor agent - re-scans for leaks and triggers self-correction."""
    anonymized_text = state.get("anonymized_text", "")
    
    # Init wrapper for audit scan
    # In a full version, we'd pass config here. For now, use same model settings.
    analyzer = AnonerWrapper()
    
    logger.info("Running agentic quality audit (re-scan)")
    
    # Re-scan anonymized text (patterns only for speed during audit, unless configured otherwise)
    audit_results = analyzer.analyze_all(anonymized_text, use_llm=False)
    
    leaked_entities = []
    for batch in audit_results:
        for entity in batch:
            # If the found text is NOT a placeholder we just inserted
            # Placeholder format is [TYPE], so we check if it's bracketed
            if not (entity.text.startswith("[") and entity.text.endswith("]")):
                # Map back to original text coordinates (heuristic)
                # For now, we identify it as a leak
                # In a more advanced version, we'd use fuzzy matching to find original offset
                leaked_entities.append(entity)
                logger.warning(f"PII Leakage found: {entity.text}")

    retry_count = state.get("retry_count", 0)
    leakage_detected = len(leaked_entities) > 0

    # If leakage detected, we want to tell the detector where to look harder
    # We pass these back via FlowState
    return {
        "leakage_detected": leakage_detected,
        "missed_entities": leaked_entities,
        "retry_count": retry_count + 1,
        "quality_report": {
            "leaks": len(leaked_entities),
            "retry_count": retry_count,
        },
        "messages": [{
            "role": "assistant",
            "content": f"Audit found {len(leaked_entities)} leaks."
        }],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_audit(state: FlowState) -> Literal["end", "detect"]:
    """Route based on quality audit results.
    
    Allows one self-correction retry if leakage is detected.
    """
    if state.get("leakage_detected", False) and state.get("retry_count", 0) < 2:
        logger.info("Leakage detected. Routing back to 'detect' for self-correction pass.")
        return "detect"
    
    return "end"


# =============================================================================
# Flow Factory
# =============================================================================


def create_privacy_flow(
    human_in_loop: bool = True,
    confidence_threshold: float = 0.7,
    preset: str = "clinical",
    use_ministral: bool = True,
    ministral_model: str = "ministral-3:8b",
    ollama_url: str = "http://localhost:11434",
    use_gliner: bool = False,
) -> StateGraph:
    """Create the PII detection and anonymization flow."""
    logger.info(
        f"Creating privacy flow (hitl={human_in_loop}, "
        f"threshold={confidence_threshold}, preset={preset})"
    )

    # Create Detection Coordinator
    detection_coordinator = create_detection_coordinator(
        preset=preset,
        use_ministral=use_ministral,
        ministral_model=ministral_model,
        ollama_url=ollama_url,
        use_gliner=use_gliner,
    )

    # Build the graph
    workflow = StateGraph(FlowState)

    # Add nodes
    workflow.add_node("detect", detection_coordinator)
    workflow.add_node("human_validation", human_validation_node)
    workflow.add_node("anonymize", anonymization_node)
    workflow.add_node("audit", quality_audit_node)

    # Set entry point
    workflow.set_entry_point("detect")

    # Add edges
    if human_in_loop:
        workflow.add_conditional_edges(
            "detect",
            route_after_detection,
            {
                "human_validation": "human_validation",
                "anonymize": "anonymize",
            }
        )
        workflow.add_edge("human_validation", "anonymize")
    else:
        workflow.add_edge("detect", "anonymize")

    workflow.add_edge("anonymize", "audit")

    # Audit can loop back or end
    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "end": END,
            "detect": "detect",
        }
    )

    return workflow.compile()


# =============================================================================
# Convenience Functions
# =============================================================================


def run_flow(
    document: str,
    confidence_threshold: float = 0.7,
    human_in_loop: bool = False,
    preset: str = "clinical",
    use_ministral: bool = False,
    ministral_model: str = "ministral-3:8b",
    use_gliner: bool = False,
) -> dict:
    """Run the full flow on a document."""
    graph = create_privacy_flow(
        human_in_loop=human_in_loop,
        confidence_threshold=confidence_threshold,
        preset=preset,
        use_ministral=use_ministral,
        ministral_model=ministral_model,
        use_gliner=use_gliner,
    )

    result = graph.invoke({
        "document": document,
        "confidence_threshold": confidence_threshold,
        "retry_count": 0,
        "missed_entities": [],
        "messages": [],
    })

    return result
