"""Reflective Resolver Node.

This node resolves overlapping entity detections using a hybrid approach:
1. Fast Path: Uses confidence scores and heuristic specificity for clear cases.
2. Agentic Path: Uses an LLM to resolve ambiguous cases based on context.

It replaces the purely heuristic OverlapResolver.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

from components.overlap_resolver import OverlapResolver, ResolutionStrategy
from wrappers.llm_wrapper import PromptManagedLLM

logger = logging.getLogger(__name__)


class ReflectiveResolver(OverlapResolver):
    """Context-aware overlap resolver with LLM support."""

    def __init__(
        self,
        strategy: str = ResolutionStrategy.HYBRID.value,
        llm: Optional[PromptManagedLLM] = None,
        ambiguity_threshold: float = 0.15,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
    ):

        """Initialize the reflective resolver.

        Args:
            strategy: Base resolution strategy
            llm: LLM wrapper for ambiguous reasoning
            ambiguity_threshold: Score difference below which to call LLM
            ollama_url: URL for direct Ollama connection (if llm is None)
            model: Model name for direct Ollama connection
        """
        super().__init__(strategy=strategy)
        self.llm = llm
        self.ollama_config = None
        
        if ollama_url and model:
            self.ollama_config = {"url": ollama_url, "model": model}
            
        self.ambiguity_threshold = ambiguity_threshold

    def resolve(self, entities: List[Any], text: str) -> List[Any]:
        """Resolve overlaps using reflection for ambiguous cases.

        Args:
            entities: List of DetectedEntity objects
            text: Original document text (needed for context)

        Returns:
            Resolved list of non-overlapping entities
        """
        if not entities:
            return []

        # Find cliques of overlapping entities
        cliques = self._build_overlap_cliques(entities)
        resolved = []

        for clique in cliques:
            if len(clique) == 1:
                resolved.append(clique[0])
            else:
                winner = self._resolve_clique_reflectively(clique, text)
                resolved.append(winner)

        return sorted(resolved, key=lambda e: e.start)

    def _resolve_clique_reflectively(
        self,
        clique: List[Any],
        text: str
    ) -> Any:
        """Resolve a single conflict clique."""
        
        # 1. Check for Super/Sub set relationships first (Fast path)
        # If one entity is fully contained in another, and they are compatible,
        # usually prefer the longer one (e.g. "Berlin" vs "Berlin, Germany")
        # unless specificity dictates otherwise.
        
        # 2. Check scores
        sorted_clique = sorted(clique, key=lambda e: e.score, reverse=True)
        top_1 = sorted_clique[0]
        top_2 = sorted_clique[1]

        score_diff = top_1.score - top_2.score

        # Fast Path: Significant score difference
        if score_diff > self.ambiguity_threshold:
            return super()._resolve_clique(clique)

        # Agentic Path: Ambiguous case
        if self.llm or self.ollama_config:
            try:
                logger.info(f"Ambiguous overlap ({score_diff:.3f}): {top_1.entity_type} vs {top_2.entity_type}. Asking LLM.")
                winner = self._ask_llm_resolution(clique, text)
                if winner:
                    return winner
            except Exception as e:
                logger.warning(f"Reflective resolution failed: {e}. Falling back to heuristics.")

        # Fallback to heuristics
        return super()._resolve_clique(clique)

    def _ask_llm_resolution(self, clique: List[Any], text: str) -> Optional[Any]:
        """Ask LLM to resolve the conflict."""
        
        # Extract context (window around the entity)
        start = min(e.start for e in clique)
        end = max(e.end for e in clique)
        ctx_start = max(0, start - 50)
        ctx_end = min(len(text), end + 50)
        context = text[ctx_start:ctx_end]
        
        # Format candidates
        candidates_desc = []
        for i, e in enumerate(clique):
            candidates_desc.append(f"{i}: {e.entity_type} ('{e.text}')")
            
        candidates_str = "\n".join(candidates_desc)

        prompt = f"""Analyze the following text snippet from a German clinical document:
" ...{context}... "

There is a conflict in PII detection. Which entity type is correct for the span in the middle?

Candidates:
{candidates_str}

Select the single best candidate index (0, 1, etc) based on the context.
Return ONLY the index number.
"""
        # Call LLM
        
        # Option 1: Use Wrapper
        if self.llm:
            try:
                recognizer = self.llm.create_recognizer()
                # Assuming recognizer has a way to generate text or we access internal engine
                # OllamaNERecognizer (from presidio-analyzer custom) doesn't expose raw generation easily
                # typically. But we can import the engine if needed, or rely on wrapper method if exists.
                # For now, let's assume we can access llm_engine if available or use requests directly.
                if hasattr(recognizer, "llm_engine"):
                     response = recognizer.llm_engine.generate_text(prompt)
                else: 
                     # Fallback to direct request if wrapper doesn't expose engine
                     import requests
                     payload = {
                        "model": self.llm.model,
                        "prompt": prompt,
                        "stream": False
                     }
                     r = requests.post(f"{self.llm.ollama_url}/api/generate", json=payload)
                     response = r.json().get("response", "")
            except Exception as e:
                logger.warning(f"Wrapper LLM call failed: {e}")
                return None
                
        # Option 2: Use direct config
        elif self.ollama_config:
            try:
                import requests
                payload = {
                    "model": self.ollama_config["model"],
                    "prompt": prompt,
                    "stream": False
                }
                r = requests.post(f"{self.ollama_config['url']}/api/generate", json=payload)
                response = r.json().get("response", "")
            except Exception as e:
                logger.warning(f"Direct LLM call failed: {e}")
                return None
        else:
            return None

        
        try:
            # Parse simple index
            idx = int(response.strip().split()[0])
            if 0 <= idx < len(clique):
                return clique[idx]
        except (ValueError, IndexError):
            pass
            
        return None
