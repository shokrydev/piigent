"""Detection Coordinator - Orchestrates PII detection using Presidio recognizers.

Imports directly from the custom Presidio fork (anoner):
- German pattern recognizers (country_specific/germany/)
- OllamaNERecognizer (ner/)
- NvidiaGLiNERPIIRecognizer (ner/)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Literal, Optional

from presidio_analyzer.predefined_recognizers import (
    # German healthcare recognizers
    DeKvnrRecognizer,
    DeLanrRecognizer,
    DeBsnrRecognizer,
    DeTelematikIdRecognizer,
    # Identity documents
    DePersonalIdRecognizer,
    DePassportRecognizer,
    DeDriverLicenseRecognizer,
    # Other identifiers
    DeTaxIdRecognizer,
    DeSocialSecurityRecognizer,
    DePostalCodeRecognizer,
    # Business
    DeCommercialRegisterRecognizer,
    DeVatCodeRecognizer,
    DeLicensePlateRecognizer,
)

from graph.state import FlowState, DetectedEntity
from agents.helpers.aggregator import EntityAggregator
from components.reflective_resolution import ReflectiveResolver

logger = logging.getLogger(__name__)

# Timeout for LLM/NER recognizers
LLM_TIMEOUT = 30

# Recognizer presets
PRESETS = {
    "clinical": [
        DeKvnrRecognizer,
        DeLanrRecognizer,
        DeBsnrRecognizer,
        DeTelematikIdRecognizer,
        DePersonalIdRecognizer,
        DeSocialSecurityRecognizer,
        DePostalCodeRecognizer,
    ],
    "full_german": [
        DeKvnrRecognizer,
        DeLanrRecognizer,
        DeBsnrRecognizer,
        DeTelematikIdRecognizer,
        DePersonalIdRecognizer,
        DePassportRecognizer,
        DeDriverLicenseRecognizer,
        DeTaxIdRecognizer,
        DeSocialSecurityRecognizer,
        DePostalCodeRecognizer,
        DeCommercialRegisterRecognizer,
        DeVatCodeRecognizer,
        DeLicensePlateRecognizer,
    ],
    "minimal": [
        DeKvnrRecognizer,
        DePostalCodeRecognizer,
    ],
}


def _run_german_recognizers(text: str, preset: str = "clinical") -> List[DetectedEntity]:
    """Run German pattern recognizers from Presidio fork."""
    recognizer_classes = PRESETS.get(preset, PRESETS["clinical"])
    entities = []

    for cls in recognizer_classes:
        try:
            recognizer = cls()
            results = recognizer.analyze(
                text=text,
                entities=recognizer.get_supported_entities(),
                nlp_artifacts=None,
            )
            for r in results:
                entities.append(DetectedEntity(
                    entity_type=r.entity_type,
                    text=text[r.start:r.end],
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    recognizer="german",
                ))
        except Exception as e:
            logger.warning(f"Recognizer {cls.__name__} failed: {e}")

    return entities


def _run_ministral(text: str, model: str, ollama_url: str) -> List[DetectedEntity]:
    """Run Ministral LLM recognizer from Presidio fork."""
    try:
        from presidio_analyzer.predefined_recognizers import OllamaNERecognizer

        recognizer = OllamaNERecognizer(
            ollama_url=ollama_url,
            model=model,
            supported_language="de",
            timeout=LLM_TIMEOUT,
        )
        recognizer.load()

        results = recognizer.analyze(
            text=text,
            entities=["PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER",
                      "EMAIL_ADDRESS", "DATE_TIME", "AGE", "IBAN", "ID"],
        )
        return [
            DetectedEntity(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
                recognizer="ministral",
            )
            for r in results
        ]
    except ImportError:
        logger.warning("OllamaNERecognizer not available")
        return []
    except Exception as e:
        logger.warning(f"Ministral failed: {e}")
        return []


def _run_gliner(text: str) -> List[DetectedEntity]:
    """Run GLiNER recognizer from Presidio fork."""
    try:
        from presidio_analyzer.predefined_recognizers import NvidiaGLiNERPIIRecognizer

        recognizer = NvidiaGLiNERPIIRecognizer(supported_language="de")
        recognizer.load()

        results = recognizer.analyze(
            text=text,
            entities=["PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER",
                      "EMAIL_ADDRESS", "DATE_TIME", "AGE", "ID"],
        )
        return [
            DetectedEntity(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
                recognizer="gliner",
            )
            for r in results
        ]
    except ImportError:
        logger.warning("NvidiaGLiNERPIIRecognizer not available")
        return []
    except Exception as e:
        logger.warning(f"GLiNER failed: {e}")
        return []


class DetectionCoordinator:
    """Coordinates parallel PII detection using Presidio recognizers."""

    def __init__(
        self,
        preset: str = "clinical",
        use_ministral: bool = False,
        ministral_model: str = "ministral",
        ollama_url: str = "http://localhost:11434",
        use_gliner: bool = False,
    ):
        self.preset = preset
        self.use_ministral = use_ministral
        self.ministral_model = ministral_model
        self.ollama_url = ollama_url
        self.use_gliner = use_gliner
        
        # Initialize Agentic Resolver
        resolver = None
        if use_ministral:
             resolver = ReflectiveResolver(
                 ollama_url=ollama_url,
                 model=ministral_model
             )
        
        self.aggregator = EntityAggregator(resolver=resolver)

        logger.info(f"DetectionCoordinator(preset={preset}, ministral={use_ministral}, gliner={use_gliner})")

    def __call__(self, state: FlowState) -> dict:
        """LangGraph node - execute detection flow."""
        document = state["document"]
        threshold = state.get("confidence_threshold", 0.7)

        logger.info(f"Detecting PII in {len(document)} chars")

        results = []
        recognizers_used = []

        # Run recognizers (parallel if using LLM/NER)
        if self.use_ministral or self.use_gliner:
            with ThreadPoolExecutor(max_workers=3) as executor:
                german_future = executor.submit(_run_german_recognizers, document, self.preset)

                ministral_future = None
                if self.use_ministral:
                    ministral_future = executor.submit(
                        _run_ministral, document, self.ministral_model, self.ollama_url
                    )

                gliner_future = None
                if self.use_gliner:
                    gliner_future = executor.submit(_run_gliner, document)

                # Collect results
                results.append(german_future.result(timeout=10))
                recognizers_used.append("german")

                if ministral_future:
                    try:
                        r = ministral_future.result(timeout=LLM_TIMEOUT)
                        if r:
                            results.append(r)
                            recognizers_used.append("ministral")
                    except TimeoutError:
                        logger.warning("Ministral timed out")

                if gliner_future:
                    try:
                        r = gliner_future.result(timeout=LLM_TIMEOUT)
                        if r:
                            results.append(r)
                            recognizers_used.append("gliner")
                    except TimeoutError:
                        logger.warning("GLiNER timed out")
        else:
            results.append(_run_german_recognizers(document, self.preset))
            recognizers_used.append("german")

        # Aggregate
        entities = self.aggregator.aggregate(results, text=document)
        min_conf = min((e.score for e in entities), default=1.0)
        needs_validation = min_conf < threshold

        # Summary
        type_counts = {}
        for e in entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

        message = (
            f"Found {len(entities)} entities. "
            f"Types: {type_counts}. "
            f"Min confidence: {min_conf:.2f}. "
            f"Recognizers: {recognizers_used}."
        )
        logger.info(message)

        return {
            "detected_entities": entities,
            "min_confidence": min_conf,
            "needs_human_validation": needs_validation,
            "messages": [{"role": "assistant", "content": message}],
        }


def create_detection_coordinator(
    preset: str = "clinical",
    use_ministral: bool = False,
    ministral_model: str = "ministral",
    ollama_url: str = "http://localhost:11434",
    use_gliner: bool = False,
    **kwargs,  # Ignore deprecated args
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
