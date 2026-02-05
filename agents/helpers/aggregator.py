"""Aggregation logic for merging results from multiple recognizers."""

import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Literal, Optional

from graph.state import DetectedEntity
from components.reflective_resolution import ReflectiveResolver

logger = logging.getLogger(__name__)

AggregationStrategy = Literal["max", "boost"]


class EntityAggregator:
    """Merge and deduplicate results from multiple recognizers."""

    def __init__(self, resolver: Optional[ReflectiveResolver] = None):
        self.resolver = resolver

    @staticmethod
    def _entities_overlap(e1: DetectedEntity, e2: DetectedEntity) -> bool:
        """Check if two entities overlap in position."""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    @staticmethod
    def _merge_overlapping(
        entities: List[DetectedEntity], strategy: AggregationStrategy
    ) -> List[DetectedEntity]:
        """Merge overlapping entities, keeping the best one."""
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x.start, -x.score))
        merged = []

        for entity in sorted_entities:
            # Check if this entity overlaps with any existing merged entity
            overlap_found = False
            for i, existing in enumerate(merged):
                if EntityAggregator._entities_overlap(entity, existing):
                    overlap_found = True
                    # Keep the one with higher score, or boost if same type
                    if entity.entity_type == existing.entity_type:
                        if strategy == "boost" and entity.recognizer != existing.recognizer:
                            # Different recognizers agree - boost confidence
                            boosted = deepcopy(existing)
                            boosted.score = min(1.0, existing.score * 1.15)
                            merged[i] = boosted
                            logger.debug(
                                f"Boosted entity '{existing.text}' from {existing.score:.2f} to {boosted.score:.2f}"
                            )
                        elif entity.score > existing.score:
                            merged[i] = entity
                    elif entity.score > existing.score:
                        # Different types - keep higher score
                        merged[i] = entity
                    break

            if not overlap_found:
                merged.append(deepcopy(entity))

        return merged

    def aggregate(
        self,
        entity_lists: List[List[DetectedEntity]],
        strategy: AggregationStrategy = "boost",
        text: Optional[str] = None,
    ) -> List[DetectedEntity]:
        """Aggregate entities from multiple recognizers.

        Args:
            entity_lists: List of entity lists from different recognizers.
            strategy: Aggregation strategy:
                - "max": Keep highest confidence per position
                - "boost": Boost score when multiple recognizers agree

        Returns:
            Deduplicated and merged list of entities.
        """
        # Flatten all entities
        all_entities = [e for entities in entity_lists for e in entities]

        if not all_entities:
            return []

        # Group by exact position and type first
        by_position = defaultdict(list)
        for entity in all_entities:
            key = (entity.start, entity.end, entity.entity_type)
            by_position[key].append(entity)

        # Handle exact matches
        exact_merged = []
        for key, entities in by_position.items():
            if len(entities) == 1:
                exact_merged.append(entities[0])
            else:
                # Multiple recognizers found exact same entity
                best = max(entities, key=lambda x: x.score)
                if strategy == "boost":
                    boosted = deepcopy(best)
                    boosted.score = min(1.0, best.score * 1.15)
                    exact_merged.append(boosted)
                else:
                    exact_merged.append(best)

        # Now handle overlapping entities
        if self.resolver and text:
             # Use agentic resolution
             # ReflectiveResolver expects list of entities and text
             return self.resolver.resolve(exact_merged, text)
        else:
             # Fallback to heuristic merge
             result = self._merge_overlapping(exact_merged, strategy)

        # Sort by position
        result.sort(key=lambda x: x.start)

        logger.info(
            f"Aggregated {len(all_entities)} entities from {len(entity_lists)} recognizers into {len(result)} unique entities"
        )
        return result
