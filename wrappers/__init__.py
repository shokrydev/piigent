"""Wrappers for integrating PIIgent components with external systems.

This module provides adapters and wrappers for:
- LLM recognizers with prompt genome management
- SynPII with curriculum-based configuration
"""

from wrappers.llm_wrapper import PromptManagedLLM, GenomeEvaluator
from wrappers.synpii_wrapper import CurriculumSynPIIWrapper, GeneratedDocument

__all__ = [
    "PromptManagedLLM",
    "GenomeEvaluator",
    "CurriculumSynPIIWrapper",
    "GeneratedDocument",
]
