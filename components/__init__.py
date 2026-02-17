"""Multi-Step NER Flow.

This module provides a sophisticated multi-step NER flow that:
1. Parses documents into semantic sections
2. Applies section-aware entity expectations
3. Applies domain-specific rules
4. Resolves overlapping entities
5. Validates consistency
"""

from components.section_parser import SectionParser, ClinicalSection
from components.domain_rules import DomainRulesEngine, Rule, RuleAction
from components.reflective_resolution import OverlapResolver, ResolutionStrategy
from components.consistency import ConsistencyValidator

__all__ = [
    "SectionParser",
    "ClinicalSection",
    "DomainRulesEngine",
    "Rule",
    "RuleAction",
    "OverlapResolver",
    "ResolutionStrategy",
    "ConsistencyValidator",
]
