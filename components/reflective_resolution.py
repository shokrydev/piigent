"""Reflective Resolver Agent.

This agent resolves overlapping entity detections using a hybrid approach:
1. Fast Path: Uses confidence scores and heuristic specificity for clear cases.
2. Agentic Path: Uses an LLM to resolve ambiguous cases based on context.

It replaces the purely heuristic OverlapResolver.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Any

# =============================================================================
# Overlap Resolution Base
# =============================================================================

class ResolutionStrategy(str, Enum):
    """Strategies for resolving overlapping entities."""
    PREFER_SPECIFIC = "prefer_specific"      # Keep more specific types
    PREFER_LONGER = "prefer_longer"          # Keep longer spans
    PREFER_SHORTER = "prefer_shorter"        # Keep shorter spans
    PREFER_CONFIDENCE = "prefer_confidence"  # Keep higher confidence
    MERGE = "merge"                          # Combine into one entity
    KEEP_ALL = "keep_all"                    # Keep all overlapping entities
    HYBRID = "hybrid"                        # Fast path + Agentic path


# Type specificity scores (higher = more specific, should be preferred)
TYPE_SPECIFICITY = {
    # German-specific identifiers (very specific)
    "DE_KVNR": 10,
    "DE_LANR": 10,
    "DE_BSNR": 10,
    "DE_TELEMATIK_ID": 10,
    "DE_PERSONAL_ID": 10,
    "DE_TAX_ID": 9,
    "DE_SOCIAL_SECURITY": 9,
    "DE_PASSPORT": 9,
    "DE_DRIVER_LICENSE": 9,
    "DE_COMMERCIAL_REGISTER": 9,
    "DE_VAT_CODE": 9,
    "DE_LICENSE_PLATE": 9,
    "DE_POSTAL_CODE": 8,

    # Standard identifiers
    "CREDIT_CARD": 9,
    "IBAN": 9,
    "EMAIL_ADDRESS": 8,
    "PHONE_NUMBER": 7,

    # Semi-specific types
    "AGE": 6,
    "OCCUPATION": 6,
    "DATE_TIME": 5,

    # Generic types (less specific)
    "PERSON": 4,
    "ORGANIZATION": 3,
    "LOCATION": 2,

    # Catch-all
    "NRP": 1,  # Non-recognized pattern
}


@dataclass
class OverlapConflict:
    """Record of an overlap conflict and its resolution."""
    entity_a: Dict
    entity_b: Dict
    overlap_ratio: float
    resolution: str  # "keep_a", "keep_b", "keep_both", "merged"
    reason: str


class OverlapResolver:
    """Resolver for overlapping entity detections.

    When multiple entities overlap, the resolver applies a configurable
    strategy to decide which to keep.

    Example:
        resolver = OverlapResolver(strategy=ResolutionStrategy.PREFER_SPECIFIC)

        entities = [
            {"text": "10117 Berlin", "entity_type": "LOCATION", "start": 0, "end": 12},
            {"text": "10117", "entity_type": "DE_POSTAL_CODE", "start": 0, "end": 5},
        ]

        resolved = resolver.resolve(entities)
        # Keeps both: postal code is more specific for its span
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.PREFER_SPECIFIC,
        overlap_threshold: float = 0.5,  # Minimum overlap ratio to consider conflict
        custom_specificity: Optional[Dict[str, int]] = None,
    ):
        """Initialize the overlap resolver.

        Args:
            strategy: Strategy for resolving overlaps
            overlap_threshold: Minimum overlap ratio to consider as conflict
            custom_specificity: Custom type specificity scores
        """
        self.strategy = strategy
        self.overlap_threshold = overlap_threshold

        self.specificity = dict(TYPE_SPECIFICITY)
        if custom_specificity:
            self.specificity.update(custom_specificity)

        self.conflicts: List[OverlapConflict] = []

    def _get_val(self, item, key, default=None):
        """Get value from dict or object."""
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)


    def resolve(self, entities: List[Dict]) -> List[Dict]:
        """Resolve overlapping entities.

        Args:
            entities: List of entity dicts with start, end, entity_type, score

        Returns:
            List of entities with overlaps resolved
        """
        if len(entities) <= 1:
            return entities

        # Sort by start position, then by end position (longer first)
        sorted_entities = sorted(entities, key=lambda e: (self._get_val(e, "start"), -self._get_val(e, "end")))

        # Track which entities to keep
        keep = [True] * len(sorted_entities)

        # Compare all pairs
        for i in range(len(sorted_entities)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(sorted_entities)):
                if not keep[j]:
                    continue

                overlap = self._calculate_overlap(sorted_entities[i], sorted_entities[j])

                if overlap >= self.overlap_threshold:
                    # Resolve conflict
                    resolution = self._resolve_pair(
                        sorted_entities[i],
                        sorted_entities[j],
                        overlap,
                    )

                    if resolution == "keep_a":
                        keep[j] = False
                    elif resolution == "keep_b":
                        keep[i] = False
                        break  # Entity i is removed, stop comparing
                    # "keep_both" keeps both

        return [e for e, k in zip(sorted_entities, keep) if k]

    def _calculate_overlap(self, entity_a: Dict, entity_b: Dict) -> float:
        """Calculate overlap ratio between two entities."""
        start_a, end_a = self._get_val(entity_a, "start"), self._get_val(entity_a, "end")
        start_b, end_b = self._get_val(entity_b, "start"), self._get_val(entity_b, "end")

        # Calculate overlap
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_len = overlap_end - overlap_start
        min_len = min(end_a - start_a, end_b - start_b)

        return overlap_len / min_len if min_len > 0 else 0.0

    def _resolve_pair(
        self,
        entity_a: Dict,
        entity_b: Dict,
        overlap: float,
    ) -> str:
        """Resolve conflict between two overlapping entities.

        Returns:
            "keep_a", "keep_b", or "keep_both"
        """
        type_a = self._get_val(entity_a, "entity_type", self._get_val(entity_a, "type", ""))
        type_b = self._get_val(entity_b, "entity_type", self._get_val(entity_b, "type", ""))
        score_a = self._get_val(entity_a, "score", 0.5)
        score_b = self._get_val(entity_b, "score", 0.5)
        len_a = self._get_val(entity_a, "end") - self._get_val(entity_a, "start")
        len_b = self._get_val(entity_b, "end") - self._get_val(entity_b, "start")

        resolution = "keep_both"
        reason = ""

        if self.strategy == ResolutionStrategy.PREFER_SPECIFIC:
            spec_a = self.specificity.get(type_a, 0)
            spec_b = self.specificity.get(type_b, 0)

            # If one entity fully contains the other
            if self._contains(entity_a, entity_b):
                # Keep both if specific type is subset
                if spec_b > spec_a:
                    resolution = "keep_both"
                    reason = f"{type_b} is more specific than {type_a}"
                else:
                    resolution = "keep_a"
                    reason = f"Containing entity {type_a} is equally or more specific"
            elif self._contains(entity_b, entity_a):
                if spec_a > spec_b:
                    resolution = "keep_both"
                    reason = f"{type_a} is more specific than {type_b}"
                else:
                    resolution = "keep_b"
                    reason = f"Containing entity {type_b} is equally or more specific"
            else:
                # Partial overlap - keep more specific
                if spec_a > spec_b:
                    resolution = "keep_a"
                    reason = f"{type_a} (spec={spec_a}) more specific than {type_b} (spec={spec_b})"
                elif spec_b > spec_a:
                    resolution = "keep_b"
                    reason = f"{type_b} (spec={spec_b}) more specific than {type_a} (spec={spec_a})"
                else:
                    # Same specificity - use confidence as tiebreaker
                    resolution = "keep_a" if score_a >= score_b else "keep_b"
                    reason = f"Same specificity, using confidence: {score_a:.2f} vs {score_b:.2f}"

        elif self.strategy == ResolutionStrategy.PREFER_LONGER:
            if len_a > len_b:
                resolution = "keep_a"
                reason = f"Longer span: {len_a} > {len_b}"
            elif len_b > len_a:
                resolution = "keep_b"
                reason = f"Longer span: {len_b} > {len_a}"
            else:
                resolution = "keep_a" if score_a >= score_b else "keep_b"
                reason = "Same length, using confidence as tiebreaker"

        elif self.strategy == ResolutionStrategy.PREFER_SHORTER:
            if len_a < len_b:
                resolution = "keep_a"
                reason = f"Shorter span: {len_a} < {len_b}"
            elif len_b < len_a:
                resolution = "keep_b"
                reason = f"Shorter span: {len_b} < {len_a}"
            else:
                resolution = "keep_a" if score_a >= score_b else "keep_b"
                reason = "Same length, using confidence as tiebreaker"

        elif self.strategy == ResolutionStrategy.PREFER_CONFIDENCE:
            if score_a > score_b:
                resolution = "keep_a"
                reason = f"Higher confidence: {score_a:.2f} > {score_b:.2f}"
            elif score_b > score_a:
                resolution = "keep_b"
                reason = f"Higher confidence: {score_b:.2f} > {score_a:.2f}"
            else:
                # Same confidence - use specificity as tiebreaker
                spec_a = self.specificity.get(type_a, 0)
                spec_b = self.specificity.get(type_b, 0)
                resolution = "keep_a" if spec_a >= spec_b else "keep_b"
                reason = "Same confidence, using specificity as tiebreaker"

        elif self.strategy == ResolutionStrategy.KEEP_ALL:
            resolution = "keep_both"
            reason = "Keep all strategy"

        # Log conflict
        self.conflicts.append(OverlapConflict(
            entity_a=entity_a,
            entity_b=entity_b,
            overlap_ratio=overlap,
            resolution=resolution,
            reason=reason,
        ))

        return resolution

    def _contains(self, outer: Dict, inner: Dict) -> bool:
        """Check if outer entity fully contains inner entity."""
        return self._get_val(outer, "start") <= self._get_val(inner, "start") and self._get_val(outer, "end") >= self._get_val(inner, "end")

    def _build_overlap_cliques(self, entities: List[Any]) -> List[List[Any]]:
        """Group overlapping entities into conflict cliques.

        Args:
            entities: List of DetectedEntity objects

        Returns:
            List of cliques (lists of overlapping entities)
        """
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: self._get_val(e, "start"))
        
        cliques = []
        if not sorted_entities:
            return cliques

        current_clique = [sorted_entities[0]]
        current_end = self._get_val(sorted_entities[0], "end")

        for i in range(1, len(sorted_entities)):
            entity = sorted_entities[i]
            # If this entity overlaps with the current clique's span
            if self._get_val(entity, "start") < current_end:
                current_clique.append(entity)
                current_end = max(current_end, self._get_val(entity, "end"))
            else:
                # No overlap, start a new clique
                cliques.append(current_clique)
                current_clique = [entity]
                current_end = self._get_val(entity, "end")

        cliques.append(current_clique)
        return cliques

    def _resolve_clique(self, clique: List[Any]) -> Any:
        """Resolve a conflict clique using the current strategy.

        Args:
            clique: List of overlapping entities

        Returns:
            The single entity to keep
        """
        if not clique:
            return None
        if len(clique) == 1:
            return clique[0]

        # Standard resolution: pick the best one by comparing iteratively
        winner = clique[0]
        for i in range(1, len(clique)):
            # Force high overlap to ensure resolution
            res = self._resolve_pair(winner, clique[i], 1.0)
            if res == "keep_b":
                winner = clique[i]
        return winner

    def get_conflicts(self) -> List[OverlapConflict]:
        """Get all recorded conflicts."""
        return self.conflicts

    def clear_conflicts(self) -> None:
        """Clear the conflict log."""
        self.conflicts = []

    def get_statistics(self) -> Dict:
        """Get statistics about conflict resolution."""
        resolutions = {}
        for conflict in self.conflicts:
            resolutions[conflict.resolution] = resolutions.get(conflict.resolution, 0) + 1

        return {
            "total_conflicts": len(self.conflicts),
            "resolutions": resolutions,
            "avg_overlap": (
                sum(c.overlap_ratio for c in self.conflicts) / len(self.conflicts)
                if self.conflicts else 0.0
            ),
        }

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
        
        # Call LLM via AnonerWrapper
        from wrappers.anoner_wrapper import AnonerWrapper
        
        url = self.ollama_config["url"] if self.ollama_config else self.llm.ollama_url
        model = self.ollama_config["model"] if self.ollama_config else self.llm.model
        
        wrapper = AnonerWrapper(ollama_url=url, model=model)
        response = wrapper.generate_text(prompt)

        if not response:
            return None

        
        try:
            # Parse simple index
            idx = int(response.strip().split()[0])
            if 0 <= idx < len(clique):
                return clique[idx]
        except (ValueError, IndexError):
            pass
            
        return None
