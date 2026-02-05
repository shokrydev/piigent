"""Rationale Agent for Self-Explanation.

Asks the LLM to explain its own predictions, providing
insight into why entities were tagged or missed.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Rationale:
    """An explanation for a prediction."""
    entity_text: str
    entity_type: str
    explanation: str
    confidence: float
    supporting_evidence: List[str]


class RationaleAgent:
    """Agent that explains LLM predictions.

    Generates rationales for:
    - Why an entity was tagged
    - Why an entity was missed
    - Why an entity was given a specific type

    This enables understanding model behavior and identifying
    systematic issues in detection.

    Example:
        agent = RationaleAgent()

        # Explain a prediction
        rationale = agent.explain_prediction(
            text="Herr Bauer arbeitet als Ingenieur in München.",
            entity={"text": "Herr Bauer", "entity_type": "PERSON", "start": 0, "end": 10},
        )
        print(rationale.explanation)

        # Explain a miss
        rationale = agent.explain_miss(
            text="Der 63-jährige Patient...",
            expected_entity={"text": "63-jährige", "type": "AGE", "start": 4, "end": 14},
            detected_entities=[],
        )
        print(rationale.explanation)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        timeout: float = 30.0,
    ):
        """Initialize the rationale agent.

        Args:
            ollama_url: URL of Ollama server
            model: Model to use for rationale generation
            timeout: Request timeout
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Get HTTP client for Ollama."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.ollama_url,
                    timeout=self.timeout,
                )
            except ImportError:
                logger.warning("httpx not available for RationaleAgent")
                return None
        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        client = self._get_client()
        if client is None:
            return "LLM unavailable"

        try:
            response = client.post("/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            })
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {e}"

    def explain_prediction(
        self,
        text: str,
        entity: Dict,
        context_window: int = 50,
    ) -> Rationale:
        """Explain why an entity was tagged.

        Args:
            text: Full document text
            entity: The detected entity
            context_window: Characters of context to include

        Returns:
            Rationale explaining the prediction
        """
        entity_text = entity.get("text", text[entity["start"]:entity["end"]])
        entity_type = entity.get("entity_type", entity.get("type", "UNKNOWN"))
        score = entity.get("score", 0.5)

        start = entity["start"]
        end = entity["end"]

        # Extract context
        ctx_start = max(0, start - context_window)
        ctx_end = min(len(text), end + context_window)
        context = text[ctx_start:ctx_end]

        prompt = f"""You are analyzing a PII detection decision. Explain why this text was tagged.

Entity tagged: "{entity_text}"
Entity type: {entity_type}
Confidence: {score:.2f}

Context: "...{context}..."

In 2-3 sentences, explain:
1. What patterns or features identified this as {entity_type}?
2. What context clues supported this decision?
3. How confident should we be in this classification?

Be specific and technical."""

        explanation = self._call_llm(prompt)

        # Extract supporting evidence (simple heuristic)
        evidence = []
        if "title" in explanation.lower() or "herr" in entity_text.lower() or "frau" in entity_text.lower():
            evidence.append("German title prefix")
        if "context" in explanation.lower():
            evidence.append("Contextual clues")
        if "pattern" in explanation.lower():
            evidence.append("Pattern matching")

        return Rationale(
            entity_text=entity_text,
            entity_type=entity_type,
            explanation=explanation.strip(),
            confidence=score,
            supporting_evidence=evidence,
        )

    def explain_miss(
        self,
        text: str,
        expected_entity: Dict,
        detected_entities: List[Dict],
        context_window: int = 50,
    ) -> Rationale:
        """Explain why an entity was missed.

        Args:
            text: Full document text
            expected_entity: The expected entity that was missed
            detected_entities: What was actually detected
            context_window: Characters of context

        Returns:
            Rationale explaining the miss
        """
        entity_text = expected_entity.get("text", text[expected_entity["start"]:expected_entity["end"]])
        entity_type = expected_entity.get("entity_type", expected_entity.get("type", "UNKNOWN"))

        start = expected_entity["start"]
        end = expected_entity["end"]

        # Extract context
        ctx_start = max(0, start - context_window)
        ctx_end = min(len(text), end + context_window)
        context = text[ctx_start:ctx_end]

        # Find what was detected in the same region
        overlapping = [
            e for e in detected_entities
            if not (e["end"] <= start or e["start"] >= end)
        ]
        detected_summary = ", ".join(
            f"{e.get('text', 'unknown')} as {e.get('entity_type', e.get('type', 'UNKNOWN'))}"
            for e in overlapping
        ) or "nothing"

        prompt = f"""You are analyzing why a PII entity was missed by the detection system.

Expected entity: "{entity_text}"
Expected type: {entity_type}
What was detected instead: {detected_summary}

Context: "...{context}..."

In 2-3 sentences, explain:
1. Why might this have been missed?
2. What patterns or context clues should have identified it?
3. What would help detect this in the future?

Be specific about potential improvements."""

        explanation = self._call_llm(prompt)

        return Rationale(
            entity_text=entity_text,
            entity_type=entity_type,
            explanation=explanation.strip(),
            confidence=0.0,  # It was missed
            supporting_evidence=[],
        )

    def explain_type_confusion(
        self,
        text: str,
        expected_entity: Dict,
        detected_entity: Dict,
        context_window: int = 50,
    ) -> Rationale:
        """Explain why an entity was classified as wrong type.

        Args:
            text: Full document text
            expected_entity: What was expected
            detected_entity: What was actually detected
            context_window: Characters of context

        Returns:
            Rationale explaining the confusion
        """
        entity_text = expected_entity.get("text", text[expected_entity["start"]:expected_entity["end"]])
        expected_type = expected_entity.get("entity_type", expected_entity.get("type", "UNKNOWN"))
        detected_type = detected_entity.get("entity_type", detected_entity.get("type", "UNKNOWN"))
        score = detected_entity.get("score", 0.5)

        start = expected_entity["start"]
        end = expected_entity["end"]

        ctx_start = max(0, start - context_window)
        ctx_end = min(len(text), end + context_window)
        context = text[ctx_start:ctx_end]

        prompt = f"""You are analyzing why a PII entity was classified with the wrong type.

Entity text: "{entity_text}"
Expected type: {expected_type}
Detected type: {detected_type}
Confidence: {score:.2f}

Context: "...{context}..."

In 2-3 sentences, explain:
1. Why was this classified as {detected_type} instead of {expected_type}?
2. What features caused the confusion?
3. How could we disambiguate between these types?

Be specific about distinguishing features."""

        explanation = self._call_llm(prompt)

        return Rationale(
            entity_text=entity_text,
            entity_type=f"{expected_type} (detected as {detected_type})",
            explanation=explanation.strip(),
            confidence=score,
            supporting_evidence=[f"Confused {expected_type} with {detected_type}"],
        )

    def batch_explain(
        self,
        text: str,
        expected: List[Dict],
        detected: List[Dict],
    ) -> Dict[str, List[Rationale]]:
        """Generate rationales for all predictions and errors.

        Args:
            text: Full document text
            expected: Expected entities
            detected: Detected entities

        Returns:
            Dict with 'correct', 'misses', 'confusions' rationale lists
        """
        rationales = {
            "correct": [],
            "misses": [],
            "confusions": [],
        }

        # Build lookup sets
        expected_spans = {(e["start"], e["end"], e.get("entity_type", e.get("type"))): e for e in expected}
        detected_spans = {(d["start"], d["end"], d.get("entity_type", d.get("type"))): d for d in detected}

        # Find correct predictions (explain a sample)
        correct_keys = set(expected_spans.keys()) & set(detected_spans.keys())
        for key in list(correct_keys)[:3]:  # Limit to 3
            rationales["correct"].append(
                self.explain_prediction(text, detected_spans[key])
            )

        # Find misses
        for key, exp in expected_spans.items():
            if key not in detected_spans:
                # Check if detected with wrong type
                same_span = [
                    d for d in detected
                    if d["start"] == exp["start"] and d["end"] == exp["end"]
                ]
                if same_span:
                    rationales["confusions"].append(
                        self.explain_type_confusion(text, exp, same_span[0])
                    )
                else:
                    rationales["misses"].append(
                        self.explain_miss(text, exp, detected)
                    )

        return rationales
