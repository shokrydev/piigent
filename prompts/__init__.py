"""Prompt Genome & Evolution System.

This module provides a genetic approach to prompt optimization with:
- Structured PromptGenotype representation
- Mutation operators for systematic prompt modification
- Crossover between high-performing prompts
- Fitness lineage tracking
- LLM-guided mutation proposals
"""

from prompts.genome import (
    PromptGenotype,
    PromptExample,
)
from prompts.store import PromptGenomeStore
from prompts.mutations import (
    MutationOperator,
    MutationEngine,
    MutationRecord,
)
from prompts.crossover import (
    CrossoverEngine,
    select_best_examples,
)
from prompts.fitness import (
    PromptFitnessTracker,
    FitnessRecord,
    PopulationManager,
)
from prompts.llm_mutation import (
    LLMMutationAdvisor,
    MutationProposal,
)

__all__ = [
    # Genome
    "PromptGenotype",
    "PromptExample",
    "PromptGenomeStore",
    # Mutations
    "MutationOperator",
    "MutationEngine",
    "MutationRecord",
    # Crossover
    "CrossoverEngine",
    "select_best_examples",
    # Fitness
    "PromptFitnessTracker",
    "FitnessRecord",
    "PopulationManager",
    # LLM Mutation
    "LLMMutationAdvisor",
    "MutationProposal",
]
