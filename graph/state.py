"""Shared state schema for the PIIgent flow."""

from typing import TypedDict, Annotated
from dataclasses import dataclass
import operator


@dataclass
class DetectedEntity:
    """A detected PII entity with metadata."""

    entity_type: str  # PERSON, EMAIL, PHONE, DATE, INSURANCE_ID, etc.
    text: str  # The actual PII text
    start: int  # Start position in document
    end: int  # End position in document
    score: float  # Confidence score 0.0-1.0
    recognizer: str  # Which recognizer found it

    def __repr__(self) -> str:
        return f"DetectedEntity({self.entity_type}: '{self.text}' [{self.start}:{self.end}] score={self.score:.2f})"


class FlowState(TypedDict):
    """Shared state across all flow agents."""

    # Input
    document: str
    confidence_threshold: float

    # Detection phase
    detected_entities: list[DetectedEntity]
    min_confidence: float
    needs_human_validation: bool

    # Messages for agent reasoning
    messages: Annotated[list, operator.add]

    # Anonymization phase (for later agents)
    anonymization_strategy: dict
    anonymized_text: str

    # Quality audit phase
    quality_report: dict
    leakage_detected: bool
    retry_count: int
    missed_entities: list[DetectedEntity]
