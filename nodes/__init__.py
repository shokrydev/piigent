"""Multi-Step NER Pipeline.

This module provides a sophisticated multi-step NER pipeline that:
1. Parses documents into semantic sections
2. Applies section-aware entity expectations
3. Applies domain-specific rules
4. Resolves overlapping entities
5. Validates consistency
"""

from nodes.section_parser import SectionParser, ClinicalSection
from nodes.domain_rules import DomainRulesEngine, Rule, RuleAction
from nodes.overlap_resolver import OverlapResolver, ResolutionStrategy
from nodes.consistency import ConsistencyValidator

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
