"""Prompt Mutation Operators.

Systematic operators for mutating prompt genomes,
enabling genetic-style prompt optimization.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from prompts.genome import PromptGenotype, PromptExample


class MutationOperator(str, Enum):
    """Mutation operators for prompt genomes."""
    # Example mutations
    ADD_EXAMPLE = "add_example"           # Add new few-shot example
    SWAP_EXAMPLE = "swap_example"         # Replace with better example
    REMOVE_EXAMPLE = "remove_example"     # Remove unhelpful example
    SHUFFLE_EXAMPLES = "shuffle_examples" # Reorder examples

    # Instruction mutations
    REPHRASE_INSTRUCTION = "rephrase"     # Reword core instruction
    ADD_CONSTRAINT = "add_constraint"     # Add explicit constraint
    REMOVE_CONSTRAINT = "remove"          # Remove constraint
    EMPHASIZE = "emphasize"               # Add emphasis to instruction

    # Entity-specific
    UPDATE_DEFINITION = "update_definition"  # Modify entity definition
    ADD_PATTERN = "add_pattern"              # Add pattern to definition
    REMOVE_PATTERN = "remove_pattern"        # Remove pattern

    # Structure
    REORDER_SECTIONS = "reorder"          # Reorder prompt sections


@dataclass
class MutationRecord:
    """Record of a mutation operation."""
    operator: MutationOperator
    target: str
    description: str
    parent_id: str
    child_id: str


class MutationEngine:
    """Engine for applying mutations to prompt genomes.

    Provides systematic mutation operators that can be applied
    to evolve prompts over generations.

    Example:
        engine = MutationEngine()

        # Apply a specific mutation
        new_genome = engine.apply_mutation(
            genome=current_genome,
            operator=MutationOperator.ADD_EXAMPLE,
            params={"entity_type": "OCCUPATION", "example": example},
        )

        # Apply random mutation
        new_genome = engine.random_mutation(current_genome)
    """

    def __init__(
        self,
        mutation_rate: float = 0.3,  # Probability of mutation
        example_pool: Optional[List[PromptExample]] = None,
    ):
        """Initialize the mutation engine.

        Args:
            mutation_rate: Base probability of mutation
            example_pool: Pool of examples to draw from
        """
        self.mutation_rate = mutation_rate
        self.example_pool = example_pool or []
        self.mutation_history: List[MutationRecord] = []

    def apply_mutation(
        self,
        genome: PromptGenotype,
        operator: MutationOperator,
        params: Optional[Dict] = None,
    ) -> PromptGenotype:
        """Apply a specific mutation to a genome.

        Args:
            genome: The genome to mutate
            operator: The mutation operator to apply
            params: Parameters for the mutation

        Returns:
            New mutated genome
        """
        params = params or {}
        new_genome = genome.clone()

        if operator == MutationOperator.ADD_EXAMPLE:
            self._add_example(new_genome, params)

        elif operator == MutationOperator.SWAP_EXAMPLE:
            self._swap_example(new_genome, params)

        elif operator == MutationOperator.REMOVE_EXAMPLE:
            self._remove_example(new_genome, params)

        elif operator == MutationOperator.SHUFFLE_EXAMPLES:
            self._shuffle_examples(new_genome)

        elif operator == MutationOperator.ADD_CONSTRAINT:
            self._add_constraint(new_genome, params)

        elif operator == MutationOperator.REMOVE_CONSTRAINT:
            self._remove_constraint(new_genome, params)

        elif operator == MutationOperator.UPDATE_DEFINITION:
            self._update_definition(new_genome, params)

        elif operator == MutationOperator.EMPHASIZE:
            self._emphasize(new_genome, params)

        elif operator == MutationOperator.REPHRASE_INSTRUCTION:
            self._rephrase_instruction(new_genome, params)

        # Record mutation
        self.mutation_history.append(MutationRecord(
            operator=operator,
            target=params.get("target", ""),
            description=str(params),
            parent_id=genome.id,
            child_id=new_genome.id,
        ))

        return new_genome

    def random_mutation(
        self,
        genome: PromptGenotype,
    ) -> PromptGenotype:
        """Apply a random mutation.

        Args:
            genome: The genome to mutate

        Returns:
            Mutated genome
        """
        if random.random() > self.mutation_rate:
            return genome.clone()  # No mutation

        # Choose operator based on genome state
        available_operators = self._get_available_operators(genome)
        operator = random.choice(available_operators)

        # Generate random params
        params = self._generate_random_params(operator, genome)

        return self.apply_mutation(genome, operator, params)

    def _get_available_operators(
        self,
        genome: PromptGenotype,
    ) -> List[MutationOperator]:
        """Get operators available for this genome state."""
        operators = [
            MutationOperator.ADD_CONSTRAINT,
            MutationOperator.EMPHASIZE,
        ]

        if genome.examples:
            operators.extend([
                MutationOperator.REMOVE_EXAMPLE,
                MutationOperator.SWAP_EXAMPLE,
                MutationOperator.SHUFFLE_EXAMPLES,
            ])

        if self.example_pool or genome.examples:
            operators.append(MutationOperator.ADD_EXAMPLE)

        if genome.constraints:
            operators.append(MutationOperator.REMOVE_CONSTRAINT)

        if genome.entity_definitions:
            operators.append(MutationOperator.UPDATE_DEFINITION)

        return operators

    def _generate_random_params(
        self,
        operator: MutationOperator,
        genome: PromptGenotype,
    ) -> Dict:
        """Generate random parameters for an operator."""
        params = {}

        if operator == MutationOperator.ADD_EXAMPLE:
            if self.example_pool:
                params["example"] = random.choice(self.example_pool)
            else:
                params["entity_type"] = random.choice(list(genome.entity_definitions.keys()) or ["PERSON"])

        elif operator == MutationOperator.REMOVE_EXAMPLE:
            if genome.examples:
                params["index"] = random.randint(0, len(genome.examples) - 1)

        elif operator == MutationOperator.SWAP_EXAMPLE:
            if genome.examples and self.example_pool:
                params["index"] = random.randint(0, len(genome.examples) - 1)
                params["example"] = random.choice(self.example_pool)

        elif operator == MutationOperator.REMOVE_CONSTRAINT:
            if genome.constraints:
                params["index"] = random.randint(0, len(genome.constraints) - 1)

        elif operator == MutationOperator.UPDATE_DEFINITION:
            if genome.entity_definitions:
                params["entity_type"] = random.choice(list(genome.entity_definitions.keys()))

        return params

    def _add_example(self, genome: PromptGenotype, params: Dict) -> None:
        """Add an example to the genome."""
        if "example" in params:
            genome.examples.append(params["example"])
        else:
            entity_type = params.get("entity_type", "PERSON")
            # Create a placeholder example
            example = PromptExample(
                input_text=f"Example text for {entity_type}",
                expected_output="[]",
                entity_type=entity_type,
                source="mutation",
            )
            genome.examples.append(example)
        genome.mutation_history.append(f"add_example:{params.get('entity_type', 'unknown')}")

    def _swap_example(self, genome: PromptGenotype, params: Dict) -> None:
        """Swap an example with another."""
        index = params.get("index", 0)
        if index < len(genome.examples) and "example" in params:
            genome.examples[index] = params["example"]
            genome.mutation_history.append(f"swap_example:{index}")

    def _remove_example(self, genome: PromptGenotype, params: Dict) -> None:
        """Remove an example from the genome."""
        index = params.get("index", 0)
        if index < len(genome.examples):
            removed = genome.examples.pop(index)
            genome.mutation_history.append(f"remove_example:{removed.entity_type}")

    def _shuffle_examples(self, genome: PromptGenotype) -> None:
        """Shuffle the order of examples."""
        random.shuffle(genome.examples)
        genome.mutation_history.append("shuffle_examples")

    def _add_constraint(self, genome: PromptGenotype, params: Dict) -> None:
        """Add a constraint to the genome."""
        constraint = params.get("constraint", "Be precise with entity boundaries")
        genome.constraints.append(constraint)
        genome.mutation_history.append(f"add_constraint:{constraint[:30]}")

    def _remove_constraint(self, genome: PromptGenotype, params: Dict) -> None:
        """Remove a constraint from the genome."""
        index = params.get("index", 0)
        if index < len(genome.constraints):
            removed = genome.constraints.pop(index)
            genome.mutation_history.append(f"remove_constraint:{removed[:30]}")

    def _update_definition(self, genome: PromptGenotype, params: Dict) -> None:
        """Update an entity definition."""
        entity_type = params.get("entity_type")
        new_definition = params.get("definition")

        if entity_type and new_definition:
            genome.entity_definitions[entity_type] = new_definition
            genome.mutation_history.append(f"update_definition:{entity_type}")

    def _emphasize(self, genome: PromptGenotype, params: Dict) -> None:
        """Add emphasis to the instruction block."""
        emphasis = params.get("text", "IMPORTANT: ")
        position = params.get("position", "start")

        if position == "start":
            genome.instruction_block = emphasis + genome.instruction_block
        else:
            genome.instruction_block = genome.instruction_block + " " + emphasis

        genome.mutation_history.append(f"emphasize:{position}")

    def _rephrase_instruction(self, genome: PromptGenotype, params: Dict) -> None:
        """Rephrase the instruction block."""
        # In a real implementation, this would use LLM to rephrase
        # For now, just record the mutation
        genome.mutation_history.append("rephrase_instruction")


def create_mutation_from_fix(
    fix: "ProposedFix",
) -> Tuple[MutationOperator, Dict]:
    """Create mutation operator and params from a ProposedFix.

    Args:
        fix: A proposed fix from FixProposalAgent

    Returns:
        Tuple of (operator, params)
    """
    from agents.fix_proposal import MutationOperator as FixMutation

    # Map fix mutations to genome mutations
    mutation_map = {
        FixMutation.ADD_EXAMPLE: MutationOperator.ADD_EXAMPLE,
        FixMutation.ADD_CONSTRAINT: MutationOperator.ADD_CONSTRAINT,
        FixMutation.EMPHASIZE_ENTITY: MutationOperator.UPDATE_DEFINITION,
        FixMutation.ADD_PATTERN: MutationOperator.UPDATE_DEFINITION,
        FixMutation.ADD_DISAMBIGUATION: MutationOperator.ADD_CONSTRAINT,
    }

    operator = mutation_map.get(fix.mutation, MutationOperator.ADD_CONSTRAINT)

    params = {
        "target": fix.target,
        "content": fix.content,
    }

    if fix.mutation == FixMutation.ADD_CONSTRAINT:
        params["constraint"] = fix.content

    elif fix.mutation == FixMutation.EMPHASIZE_ENTITY:
        parts = fix.target.split(".")
        if len(parts) > 1:
            params["entity_type"] = parts[-1]
            params["definition"] = fix.content

    return operator, params
