"""LangGraph pipeline for PII detection and anonymization.

Three-agent pipeline with conditional HITL routing:
1. Detection Coordinator - Runs recognizers, aggregates results
2. Anonymization Strategist - Applies anonymization techniques (placeholder)
3. Quality Auditor - Verifies no PII leakage (placeholder)

Human-in-the-loop validation is triggered when confidence < threshold.
"""

import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from graph.state import FlowState
from agents.core.detection_coordinator import (
    DetectionCoordinator,
    create_detection_coordinator,
    route_after_detection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Placeholder Nodes (to be implemented)
# =============================================================================


def human_validation_node(state: FlowState) -> dict:
    """Human-in-the-loop validation node.

    In the full implementation, this would:
    1. Present detected entities to human reviewer via UI
    2. Allow reviewer to confirm, reject, or modify entities
    3. Update entity list based on human feedback

    For now, this is a pass-through that marks validation as complete.
    """
    entities = state.get("detected_entities", [])

    logger.info(f"HITL: {len(entities)} entities awaiting human validation")

    # In full implementation: launch Gradio UI, wait for response
    # For now: pass through with a message
    return {
        "messages": [{
            "role": "human",
            "content": f"Human validated {len(entities)} entities."
        }],
        "needs_human_validation": False,  # Mark as validated
    }


def anonymization_node(state: FlowState) -> dict:
    """Anonymization Strategist node.

    In the full implementation, this would:
    1. Select anonymization technique per entity type:
       - Redaction: [REDACTED]
       - Pseudonymization: Replace with fake data
       - Encryption: Hash the value
    2. Apply anonymization to document
    3. Track which technique was used for each entity

    For now, applies simple redaction.
    """
    document = state["document"]
    entities = state.get("detected_entities", [])

    logger.info(f"Anonymizing {len(entities)} entities in document")

    # Simple redaction strategy (placeholder)
    anonymized = document
    strategy = {}

    # Sort entities by position (reverse) to preserve offsets
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        replacement = f"[{entity.entity_type}]"
        anonymized = anonymized[:entity.start] + replacement + anonymized[entity.end:]
        strategy[entity.entity_type] = strategy.get(entity.entity_type, "redaction")

    return {
        "anonymized_text": anonymized,
        "anonymization_strategy": strategy,
        "messages": [{
            "role": "assistant",
            "content": f"Anonymized {len(entities)} entities using redaction."
        }],
    }


def quality_audit_node(state: FlowState) -> dict:
    """Quality Auditor node.

    In the full implementation, this would:
    1. Re-run detection on anonymized text
    2. Check for PII leakage (entities that weren't anonymized)
    3. Measure utility preservation
    4. Generate quality report

    For now, performs basic leakage check.
    """
    anonymized_text = state.get("anonymized_text", "")
    original_entities = state.get("detected_entities", [])

    logger.info("Running quality audit on anonymized text")

    # Check if any original PII text still appears
    leakage_detected = False
    leaked_entities = []

    for entity in original_entities:
        if entity.text in anonymized_text:
            leakage_detected = True
            leaked_entities.append(entity)
            logger.warning(f"Leakage detected: {entity.entity_type} '{entity.text}'")

    quality_report = {
        "total_entities": len(original_entities),
        "leaked_entities": len(leaked_entities),
        "leakage_detected": leakage_detected,
        "anonymization_complete": not leakage_detected,
    }

    message = (
        f"Quality audit complete. "
        f"{'LEAKAGE DETECTED: ' + str(len(leaked_entities)) + ' entities.' if leakage_detected else 'No leakage detected.'}"
    )

    return {
        "quality_report": quality_report,
        "leakage_detected": leakage_detected,
        "messages": [{
            "role": "assistant",
            "content": message,
        }],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_audit(state: FlowState) -> Literal["end", "anonymize"]:
    """Route based on quality audit results.

    If leakage detected, could route back to anonymization.
    For now, always ends (single pass).
    """
    if state.get("leakage_detected", False):
        logger.warning("Leakage detected, but ending pipeline (single pass mode)")
    return "end"


# =============================================================================
# Pipeline Factory
# =============================================================================


def create_privacy_flow(
    human_in_loop: bool = True,
    confidence_threshold: float = 0.7,
    preset: str = "clinical",
    use_ministral: bool = True,
    ministral_model: str = "ministral",
    ollama_url: str = "http://localhost:11434",
    use_gliner: bool = False,
) -> StateGraph:
    """Create the PII detection and anonymization flow.

    Args:
        human_in_loop: Enable HITL validation for low-confidence entities.
        confidence_threshold: Threshold below which HITL is triggered.
        preset: German recognizer preset ('clinical', 'full_german', 'minimal').
        use_ministral: Enable Ministral LLM recognizer.
        ministral_model: Ollama model name.
        ollama_url: Ollama server URL.
        use_gliner: Enable GLiNER NER recognizer.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
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
        # Conditional routing based on confidence
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
        # Skip HITL, go directly to anonymization
        workflow.add_edge("detect", "anonymize")

    workflow.add_edge("anonymize", "audit")

    # Audit can loop back or end
    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "end": END,
            "anonymize": "anonymize",  # For future: retry on leakage
        }
    )

    # Compile
    graph = workflow.compile()

    logger.info("Privacy flow compiled successfully")
    return graph


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
    """Convenience function to run the full flow on a document.

    Args:
        document: Text document to process.
        confidence_threshold: HITL threshold.
        human_in_loop: Enable HITL (requires UI in full implementation).
        preset: German recognizer preset.
        use_ministral: Enable Ministral LLM.
        ministral_model: Ollama model name for Ministral.
        use_gliner: Enable GLiNER NER.

    Returns:
        Final flow state with all results.
    """
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
    })

    return result
