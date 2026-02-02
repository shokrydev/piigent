"""Human-in-the-Loop Interface.

Gradio-based UI for:
- Validating flagged PII detections
- Active learning labeling
- Reviewing and approving prompt changes
- Configurable autonomy levels
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Entity types for dropdown
ENTITY_TYPES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "AGE",
    "DATE_TIME",
    "OCCUPATION",
    "DE_KVNR",
    "DE_LANR",
    "DE_BSNR",
    "DE_POSTAL_CODE",
    "DE_TELEMATIK_ID",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IBAN",
    "CREDIT_CARD",
]


class HITLTrigger(str, Enum):
    """Triggers for human-in-the-loop intervention."""
    LOW_CONFIDENCE = "low_confidence"       # Score below threshold
    DISAGREEMENT = "disagreement"           # Recognizers disagree
    NOVEL_PATTERN = "novel_pattern"         # Pattern not seen before
    HIGH_STAKES = "high_stakes"             # Sensitive document type
    ACTIVE_LEARNING = "active_learning"     # Uncertainty sampling
    REGRESSION_DETECTED = "regression"      # Performance drop


class AutonomyMode(str, Enum):
    """Autonomy modes for the pipeline."""
    AUTONOMOUS = "autonomous"       # No human intervention
    SUPERVISED = "supervised"       # Human reviews flagged cases
    INTERACTIVE = "interactive"     # Human validates each decision


@dataclass
class PendingValidation:
    """A case pending human validation."""
    id: str
    text: str
    entity_text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    trigger: HITLTrigger
    context: str = ""  # Surrounding text
    recognizers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_row(self) -> List:
        """Convert to table row for display."""
        return [
            self.id,
            self.text[:50] + "..." if len(self.text) > 50 else self.text,
            self.entity_text,
            self.entity_type,
            f"{self.confidence:.2f}",
            self.trigger.value,
        ]


@dataclass
class ValidationResult:
    """Result of human validation."""
    case_id: str
    action: str  # "approve", "reject", "relabel"
    new_label: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class HITLManager:
    """Manager for human-in-the-loop validation queue.

    Maintains a queue of cases pending validation and tracks
    validation results for training data generation.
    """

    def __init__(self):
        self.pending_queue: List[PendingValidation] = []
        self.completed: List[ValidationResult] = []
        self.callbacks: Dict[str, Callable] = {}

    def add_case(self, case: PendingValidation) -> None:
        """Add a case to the validation queue."""
        self.pending_queue.append(case)
        logger.info(f"Added case {case.id} to HITL queue ({case.trigger.value})")

    def get_pending(self) -> List[PendingValidation]:
        """Get all pending cases."""
        return self.pending_queue

    def validate(
        self,
        case_id: str,
        action: str,
        new_label: Optional[str] = None,
    ) -> bool:
        """Record validation result.

        Args:
            case_id: ID of the case to validate
            action: "approve", "reject", or "relabel"
            new_label: New entity type if relabeling

        Returns:
            True if case was found and validated
        """
        for i, case in enumerate(self.pending_queue):
            if case.id == case_id:
                result = ValidationResult(
                    case_id=case_id,
                    action=action,
                    new_label=new_label,
                )
                self.completed.append(result)
                self.pending_queue.pop(i)

                # Call callbacks
                if "on_validation" in self.callbacks:
                    self.callbacks["on_validation"](case, result)

                logger.info(f"Validated case {case_id}: {action}")
                return True
        return False

    def on_validation(self, callback: Callable) -> None:
        """Register callback for when a case is validated."""
        self.callbacks["on_validation"] = callback

    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        return {
            "pending": len(self.pending_queue),
            "completed": len(self.completed),
            "approved": sum(1 for r in self.completed if r.action == "approve"),
            "rejected": sum(1 for r in self.completed if r.action == "reject"),
            "relabeled": sum(1 for r in self.completed if r.action == "relabel"),
        }


# Global HITL manager instance
_hitl_manager = HITLManager()


def get_hitl_manager() -> HITLManager:
    """Get the global HITL manager instance."""
    return _hitl_manager


def create_hitl_interface():
    """Create Gradio interface for HITL validation.

    Returns:
        Gradio Blocks interface
    """
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed. HITL interface unavailable.")
        return None

    hitl_manager = get_hitl_manager()

    with gr.Blocks(title="PIIgent HITL Validation") as interface:
        gr.Markdown("# PIIgent Human-in-the-Loop Validation")

        with gr.Tab("Pending Validations"):
            gr.Markdown("""
            Review flagged PII detections. Cases are flagged due to:
            - **Low confidence**: Model is uncertain
            - **Disagreement**: Multiple recognizers disagree
            - **Novel pattern**: Pattern not seen before
            """)

            refresh_btn = gr.Button("Refresh Queue", variant="secondary")

            case_display = gr.Dataframe(
                headers=["ID", "Text", "Entity", "Type", "Confidence", "Reason"],
                interactive=False,
                label="Pending Cases",
            )

            with gr.Row():
                case_id_input = gr.Textbox(label="Case ID", placeholder="Enter case ID")
                relabel_dropdown = gr.Dropdown(
                    choices=ENTITY_TYPES,
                    label="Relabel as (optional)",
                )

            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                reject_btn = gr.Button("Reject", variant="stop")
                relabel_btn = gr.Button("Relabel", variant="secondary")

            validation_status = gr.Textbox(label="Status", interactive=False)

            def refresh_queue():
                cases = hitl_manager.get_pending()
                return [case.to_row() for case in cases]

            def approve_case(case_id):
                if hitl_manager.validate(case_id, "approve"):
                    return f"Approved case {case_id}"
                return f"Case {case_id} not found"

            def reject_case(case_id):
                if hitl_manager.validate(case_id, "reject"):
                    return f"Rejected case {case_id}"
                return f"Case {case_id} not found"

            def relabel_case(case_id, new_label):
                if not new_label:
                    return "Please select a new label"
                if hitl_manager.validate(case_id, "relabel", new_label):
                    return f"Relabeled case {case_id} to {new_label}"
                return f"Case {case_id} not found"

            refresh_btn.click(refresh_queue, outputs=[case_display])
            approve_btn.click(approve_case, inputs=[case_id_input], outputs=[validation_status])
            reject_btn.click(reject_case, inputs=[case_id_input], outputs=[validation_status])
            relabel_btn.click(
                relabel_case,
                inputs=[case_id_input, relabel_dropdown],
                outputs=[validation_status],
            )

        with gr.Tab("Active Learning"):
            gr.Markdown("""
            Label uncertain cases to improve the model.
            The system selects cases where labeling will be most informative.
            """)

            text_display = gr.Textbox(
                label="Text to label",
                lines=5,
                interactive=False,
            )

            highlight_display = gr.HighlightedText(
                label="Detected entities (click to edit)",
            )

            with gr.Row():
                entity_start = gr.Number(label="Start position")
                entity_end = gr.Number(label="End position")
                entity_type_input = gr.Dropdown(
                    choices=ENTITY_TYPES,
                    label="Entity type",
                )

            add_entity_btn = gr.Button("Add Entity", variant="primary")
            next_case_btn = gr.Button("Next Case", variant="secondary")

            annotation_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Prompt Review"):
            gr.Markdown("""
            Review and approve proposed prompt changes.
            Changes are proposed by the self-critique system based on error analysis.
            """)

            current_prompt = gr.Textbox(
                label="Current Prompt",
                lines=10,
                interactive=False,
            )

            proposed_prompt = gr.Textbox(
                label="Proposed Prompt",
                lines=10,
                interactive=False,
            )

            diff_display = gr.HighlightedText(
                label="Changes",
            )

            with gr.Row():
                accept_prompt_btn = gr.Button("Accept Change", variant="primary")
                reject_prompt_btn = gr.Button("Reject Change", variant="stop")

            prompt_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Statistics"):
            gr.Markdown("## Validation Statistics")

            stats_display = gr.JSON(label="Statistics")

            def get_stats():
                return hitl_manager.get_statistics()

            refresh_stats_btn = gr.Button("Refresh Stats")
            refresh_stats_btn.click(get_stats, outputs=[stats_display])

        with gr.Tab("Configuration"):
            gr.Markdown("## HITL Configuration")

            autonomy_mode = gr.Radio(
                choices=[mode.value for mode in AutonomyMode],
                value=AutonomyMode.SUPERVISED.value,
                label="Autonomy Mode",
            )

            confidence_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="Confidence Threshold for HITL",
            )

            save_config_btn = gr.Button("Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)

            def save_config(mode, threshold):
                return f"Saved: mode={mode}, threshold={threshold}"

            save_config_btn.click(
                save_config,
                inputs=[autonomy_mode, confidence_threshold],
                outputs=[config_status],
            )

    return interface


def flag_for_validation(
    text: str,
    entity_text: str,
    entity_type: str,
    start: int,
    end: int,
    confidence: float,
    trigger: HITLTrigger,
    recognizers: Optional[List[str]] = None,
) -> str:
    """Flag a detection for human validation.

    Args:
        text: Full document text
        entity_text: The detected entity text
        entity_type: The entity type
        start: Start position
        end: End position
        confidence: Detection confidence
        trigger: Why this was flagged
        recognizers: List of recognizers that detected this

    Returns:
        Case ID for tracking
    """
    import uuid

    case = PendingValidation(
        id=str(uuid.uuid4())[:8],
        text=text,
        entity_text=entity_text,
        entity_type=entity_type,
        start=start,
        end=end,
        confidence=confidence,
        trigger=trigger,
        context=text[max(0, start-30):min(len(text), end+30)],
        recognizers=recognizers or [],
    )

    hitl_manager = get_hitl_manager()
    hitl_manager.add_case(case)

    return case.id


def should_flag_for_hitl(
    confidence: float,
    threshold: float = 0.7,
    mode: AutonomyMode = AutonomyMode.SUPERVISED,
) -> bool:
    """Determine if a detection should be flagged for HITL.

    Args:
        confidence: Detection confidence score
        threshold: Confidence threshold
        mode: Current autonomy mode

    Returns:
        True if should be flagged
    """
    if mode == AutonomyMode.AUTONOMOUS:
        return False
    if mode == AutonomyMode.INTERACTIVE:
        return True
    # SUPERVISED mode: flag low confidence
    return confidence < threshold


if __name__ == "__main__":
    # Launch the interface for testing
    interface = create_hitl_interface()
    if interface:
        interface.launch(server_port=7860, share=False)
