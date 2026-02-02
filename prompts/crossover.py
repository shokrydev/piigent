"""Prompt Crossover Operations.

Combines high-performing prompts to create offspring
that inherit the best traits from both parents.
"""

import random
from typing import Dict, List, Optional, Tuple

from prompts.genome import PromptGenotype, PromptExample


class CrossoverEngine:
    """Engine for crossover operations between prompt genomes.

    Combines two parent genomes to create offspring that inherit
    traits from both, enabling genetic-style optimization.

    Example:
        engine = CrossoverEngine()

        # Simple crossover
        child = engine.crossover(parent_a, parent_b)

        # Fitness-weighted crossover
        child = engine.weighted_crossover(
            parent_a, parent_b,
            weight_a=0.7, weight_b=0.3,
        )
    """

    def __init__(
        self,
        crossover_rate: float = 0.7,  # Probability of crossover
    ):
        """Initialize the crossover engine.

        Args:
            crossover_rate: Probability of crossover (vs just cloning)
        """
        self.crossover_rate = crossover_rate

    def crossover(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> PromptGenotype:
        """Perform crossover between two parent genomes.

        Creates a child that inherits:
        - Instruction block from parent A
        - Entity definitions merged (preferring higher-fitness)
        - Examples combined and deduplicated
        - Constraints merged

        Args:
            parent_a: First parent genome
            parent_b: Second parent genome

        Returns:
            Child genome
        """
        if random.random() > self.crossover_rate:
            # No crossover - return clone of fitter parent
            if parent_a.overall_fitness() >= parent_b.overall_fitness():
                return parent_a.clone()
            return parent_b.clone()

        child = PromptGenotype(
            id=PromptGenotype.generate_id("cross"),
            instruction_block=self._crossover_instruction(parent_a, parent_b),
            entity_definitions=self._crossover_definitions(parent_a, parent_b),
            examples=self._crossover_examples(parent_a, parent_b),
            constraints=self._crossover_constraints(parent_a, parent_b),
            output_schema=parent_a.output_schema or parent_b.output_schema,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.generation, parent_b.generation) + 1,
            mutation_history=["crossover"],
        )

        return child

    def weighted_crossover(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
        weight_a: Optional[float] = None,
        weight_b: Optional[float] = None,
    ) -> PromptGenotype:
        """Perform fitness-weighted crossover.

        Higher-fitness parent contributes more to the child.

        Args:
            parent_a: First parent
            parent_b: Second parent
            weight_a: Weight for parent A (None = use fitness)
            weight_b: Weight for parent B (None = use fitness)

        Returns:
            Child genome
        """
        # Calculate weights from fitness if not provided
        if weight_a is None or weight_b is None:
            fitness_a = parent_a.overall_fitness() + 0.1  # Add epsilon
            fitness_b = parent_b.overall_fitness() + 0.1
            total = fitness_a + fitness_b
            weight_a = fitness_a / total
            weight_b = fitness_b / total

        child = PromptGenotype(
            id=PromptGenotype.generate_id("wcross"),
            instruction_block=self._weighted_instruction(parent_a, parent_b, weight_a),
            entity_definitions=self._weighted_definitions(parent_a, parent_b),
            examples=self._weighted_examples(parent_a, parent_b, weight_a, weight_b),
            constraints=self._weighted_constraints(parent_a, parent_b, weight_a, weight_b),
            output_schema=parent_a.output_schema if weight_a >= weight_b else parent_b.output_schema,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.generation, parent_b.generation) + 1,
            mutation_history=[f"weighted_crossover:a={weight_a:.2f}"],
        )

        return child

    def multi_parent_crossover(
        self,
        parents: List[PromptGenotype],
    ) -> PromptGenotype:
        """Crossover from multiple parents.

        Args:
            parents: List of parent genomes (at least 2)

        Returns:
            Child genome
        """
        if len(parents) < 2:
            raise ValueError("Need at least 2 parents for crossover")

        # Sort by fitness
        sorted_parents = sorted(parents, key=lambda p: -p.overall_fitness())

        # Start with best parent
        child = sorted_parents[0].clone()
        child.id = PromptGenotype.generate_id("multi")
        child.parent_ids = [p.id for p in parents]

        # Incorporate traits from other parents
        for parent in sorted_parents[1:]:
            # Add any entity definitions we're missing
            for entity_type, definition in parent.entity_definitions.items():
                if entity_type not in child.entity_definitions:
                    child.entity_definitions[entity_type] = definition

            # Add unique examples
            child_example_texts = {e.input_text for e in child.examples}
            for example in parent.examples:
                if example.input_text not in child_example_texts:
                    child.examples.append(example)
                    child_example_texts.add(example.input_text)

            # Add unique constraints
            child_constraints = set(child.constraints)
            for constraint in parent.constraints:
                if constraint not in child_constraints:
                    child.constraints.append(constraint)
                    child_constraints.add(constraint)

        child.generation = max(p.generation for p in parents) + 1
        child.mutation_history = [f"multi_parent_crossover:{len(parents)}"]

        return child

    def _crossover_instruction(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> str:
        """Crossover instruction blocks."""
        # Prefer parent with higher fitness
        if parent_a.overall_fitness() >= parent_b.overall_fitness():
            return parent_a.instruction_block
        return parent_b.instruction_block

    def _crossover_definitions(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> Dict[str, str]:
        """Crossover entity definitions."""
        merged = dict(parent_a.entity_definitions)

        for entity_type, definition in parent_b.entity_definitions.items():
            if entity_type not in merged:
                # Use B's definition if A doesn't have it
                merged[entity_type] = definition
            else:
                # Prefer definition from parent with higher fitness for this type
                fitness_a = parent_a.fitness_for_entity(entity_type)
                fitness_b = parent_b.fitness_for_entity(entity_type)
                if fitness_b > fitness_a:
                    merged[entity_type] = definition

        return merged

    def _crossover_examples(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> List[PromptExample]:
        """Crossover examples."""
        combined = []
        seen_texts = set()

        # Add from both parents, preferring higher-fitness parent
        all_examples = []
        for example in parent_a.examples:
            all_examples.append((example, parent_a.overall_fitness()))
        for example in parent_b.examples:
            all_examples.append((example, parent_b.overall_fitness()))

        # Sort by parent fitness and deduplicate
        all_examples.sort(key=lambda x: -x[1])

        for example, _ in all_examples:
            if example.input_text not in seen_texts:
                combined.append(example)
                seen_texts.add(example.input_text)

        # Limit to reasonable number
        return combined[:10]

    def _crossover_constraints(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> List[str]:
        """Crossover constraints."""
        # Merge constraints, preferring from fitter parent
        merged = list(parent_a.constraints)
        seen = set(merged)

        for constraint in parent_b.constraints:
            if constraint not in seen:
                merged.append(constraint)
                seen.add(constraint)

        return merged

    def _weighted_instruction(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
        weight_a: float,
    ) -> str:
        """Select instruction based on weight."""
        if weight_a >= 0.5:
            return parent_a.instruction_block
        return parent_b.instruction_block

    def _weighted_definitions(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
    ) -> Dict[str, str]:
        """Merge definitions weighted by per-entity fitness."""
        merged = {}

        all_types = set(parent_a.entity_definitions.keys()) | set(parent_b.entity_definitions.keys())

        for entity_type in all_types:
            fitness_a = parent_a.fitness_for_entity(entity_type)
            fitness_b = parent_b.fitness_for_entity(entity_type)

            if entity_type in parent_a.entity_definitions and entity_type in parent_b.entity_definitions:
                # Both have it - use fitness to decide
                if fitness_a >= fitness_b:
                    merged[entity_type] = parent_a.entity_definitions[entity_type]
                else:
                    merged[entity_type] = parent_b.entity_definitions[entity_type]
            elif entity_type in parent_a.entity_definitions:
                merged[entity_type] = parent_a.entity_definitions[entity_type]
            else:
                merged[entity_type] = parent_b.entity_definitions[entity_type]

        return merged

    def _weighted_examples(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
        weight_a: float,
        weight_b: float,
    ) -> List[PromptExample]:
        """Select examples weighted by parent fitness."""
        combined = []
        seen = set()

        # Number of examples to take from each parent
        total_examples = len(parent_a.examples) + len(parent_b.examples)
        target_count = min(10, total_examples)

        num_from_a = int(target_count * weight_a)
        num_from_b = target_count - num_from_a

        for example in parent_a.examples[:num_from_a]:
            if example.input_text not in seen:
                combined.append(example)
                seen.add(example.input_text)

        for example in parent_b.examples[:num_from_b]:
            if example.input_text not in seen:
                combined.append(example)
                seen.add(example.input_text)

        return combined

    def _weighted_constraints(
        self,
        parent_a: PromptGenotype,
        parent_b: PromptGenotype,
        weight_a: float,
        weight_b: float,
    ) -> List[str]:
        """Select constraints weighted by parent fitness."""
        merged = []
        seen = set()

        # Start with higher-weighted parent's constraints
        if weight_a >= weight_b:
            for c in parent_a.constraints:
                merged.append(c)
                seen.add(c)
            for c in parent_b.constraints:
                if c not in seen:
                    merged.append(c)
        else:
            for c in parent_b.constraints:
                merged.append(c)
                seen.add(c)
            for c in parent_a.constraints:
                if c not in seen:
                    merged.append(c)

        return merged


def select_best_examples(
    examples_a: List[PromptExample],
    examples_b: List[PromptExample],
    max_count: int = 10,
) -> List[PromptExample]:
    """Select best examples from two lists.

    Prefers:
    1. High-confidence examples (source="human_labeled" > "high_confidence" > "manual")
    2. Diverse entity type coverage
    3. Shorter, cleaner examples

    Args:
        examples_a: First example list
        examples_b: Second example list
        max_count: Maximum examples to return

    Returns:
        Selected best examples
    """
    source_priority = {
        "human_labeled": 3,
        "high_confidence": 2,
        "manual": 1,
        "mutation": 0,
        "auto_generated": 0,
    }

    # Score each example
    scored = []
    seen_types = set()

    for example in examples_a + examples_b:
        score = source_priority.get(example.source, 0)

        # Bonus for covering new entity type
        if example.entity_type not in seen_types:
            score += 2
            seen_types.add(example.entity_type)

        # Penalty for very long examples
        if len(example.input_text) > 200:
            score -= 1

        scored.append((score, example))

    # Sort by score and deduplicate
    scored.sort(key=lambda x: -x[0])

    selected = []
    seen_texts = set()

    for _, example in scored:
        if example.input_text not in seen_texts:
            selected.append(example)
            seen_texts.add(example.input_text)

        if len(selected) >= max_count:
            break

    return selected
