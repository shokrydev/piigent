"""Anoner Wrapper - Centralizes and simplifies interactions with the Presidio fork (anoner).

Decouples PIIgent core logic from the specific import paths and initialization 
logic of the custom Presidio fork.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional, Dict

# Lazy imports for Presidio components to handle ImportError gracefully
try:
    from presidio_analyzer.predefined_recognizers import (
        # German healthcare recognizers
        DeKvnrRecognizer, DeLanrRecognizer, DeBsnrRecognizer, DeTelematikIdRecognizer,
        # Identity documents
        DePersonalIdRecognizer, DePassportRecognizer, DeDriverLicenseRecognizer,
        # Other identifiers
        DeTaxIdRecognizer, DeSocialSecurityRecognizer, DePostalCodeRecognizer,
        # Business
        DeCommercialRegisterRecognizer, DeVatCodeRecognizer, DeLicensePlateRecognizer,
    )
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

from graph.state import DetectedEntity

logger = logging.getLogger(__name__)

# Timeout for LLM/NER recognizers
LLM_TIMEOUT = 30

# Recognizer presets from anoner
PRESETS = {
    "clinical": [
        "DeKvnrRecognizer", "DeLanrRecognizer", "DeBsnrRecognizer",
        "DeTelematikIdRecognizer", "DePersonalIdRecognizer",
        "DeSocialSecurityRecognizer", "DePostalCodeRecognizer",
    ],
    "full_german": [
        "DeKvnrRecognizer", "DeLanrRecognizer", "DeBsnrRecognizer",
        "DeTelematikIdRecognizer", "DePersonalIdRecognizer",
        "DePassportRecognizer", "DeDriverLicenseRecognizer",
        "DeTaxIdRecognizer", "DeSocialSecurityRecognizer",
        "DePostalCodeRecognizer", "DeCommercialRegisterRecognizer",
        "DeVatCodeRecognizer", "DeLicensePlateRecognizer",
    ],
    "minimal": [
        "DeKvnrRecognizer", "DePostalCodeRecognizer",
    ],
}

class AnonerWrapper:
    """Wrapper for the custom Presidio fork 'anoner'."""

    def __init__(
        self,
        preset: str = "clinical",
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
    ):
        self.preset = preset
        self.ollama_url = ollama_url
        self.model = model
        
        if not PRESIDIO_AVAILABLE:
            logger.error("Presidio (anoner) is not installed. Wrappers will be non-functional.")

    def analyze_patterns(self, text: str) -> List[DetectedEntity]:
        """Run German pattern recognizers from the anoner fork."""
        if not PRESIDIO_AVAILABLE:
            return []

        # Map preset names to actual classes
        import presidio_analyzer.predefined_recognizers as pr
        recognizer_names = PRESETS.get(self.preset, PRESETS["clinical"])
        entities = []

        for name in recognizer_names:
            try:
                cls = getattr(pr, name)
                recognizer = cls()
                results = recognizer.analyze(
                    text=text,
                    entities=recognizer.get_supported_entities(),
                )
                for r in results:
                    entities.append(DetectedEntity(
                        entity_type=r.entity_type,
                        text=text[r.start:r.end],
                        start=r.start,
                        end=r.end,
                        score=r.score,
                        recognizer="german_patterns",
                    ))
            except Exception as e:
                logger.warning(f"Recognizer {name} failed: {e}")

        return entities

    def create_llm_recognizer(
        self,
        system_prompt: Optional[str] = None,
        language_additions: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        threshold: float = 0.4,
    ):
        """Factory for OllamaNERecognizer (Ollama-based NER)."""
        if not PRESIDIO_AVAILABLE:
            raise ImportError("Presidio (anoner) is not installed.")

        from presidio_analyzer.predefined_recognizers import OllamaNERecognizer
        
        recognizer = OllamaNERecognizer(
            ollama_url=self.ollama_url,
            model=self.model,
            supported_language="de",
            timeout=timeout or LLM_TIMEOUT,
            min_score=threshold,
            system_prompt=system_prompt,
            language_additions=language_additions,
        )
        recognizer.load()
        return recognizer

    def analyze_llm(self, text: str) -> List[DetectedEntity]:
        """Run Ollama-based NER (e.g., Ministral) from the anoner fork."""
        try:
            recognizer = self.create_llm_recognizer()
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
                    recognizer="llm_ner",
                ) for r in results
            ]
        except Exception as e:
            logger.warning(f"LLM NER failed: {e}")
            return []

    def generate_text(self, prompt: str) -> str:
        """Generic interface to query the LLM via Ollama API."""
        import requests
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            r = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def analyze_bert_gliner(self, text: str, model_type: str = "gliner", model_path: Optional[str] = None) -> List[DetectedEntity]:
        """Unified wrapper for BERT-like and GLiNER models in anoner."""
        if not PRESIDIO_AVAILABLE:
            return []

        try:
            from presidio_analyzer.predefined_recognizers import (
                NvidiaGLiNERPIIRecognizer, TransformersRecognizer
            )
            
            if model_type == "gliner":
                recognizer = NvidiaGLiNERPIIRecognizer(supported_language="de")
            else:
                # Placeholder for BERT/Transformers recognizer
                recognizer = TransformersRecognizer(model_path=model_path, supported_language="de")
            
            recognizer.load()
            results = recognizer.analyze(
                text=text,
                entities=["PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME", "AGE", "ID"],
            )
            return [
                DetectedEntity(
                    entity_type=r.entity_type,
                    text=text[r.start:r.end],
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    recognizer=f"local_{model_type}",
                ) for r in results
            ]
        except Exception as e:
            logger.warning(f"Local NER ({model_type}) failed: {e}")
            return []

    def analyze_all(self, text: str, use_llm: bool = False, use_ner: bool = False) -> List[List[DetectedEntity]]:
        """Orchestrate parallel detection using multiple methods."""
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "patterns": executor.submit(self.analyze_patterns, text)
            }
            if use_llm:
                futures["llm"] = executor.submit(self.analyze_llm, text)
            if use_ner:
                futures["ner"] = executor.submit(self.analyze_bert_gliner, text)

            for key, future in futures.items():
                try:
                    res = future.result(timeout=LLM_TIMEOUT + 5)
                    if res:
                        results.append(res)
                except Exception as e:
                    logger.error(f"Detection branch {key} failed: {e}")
        
        return results
