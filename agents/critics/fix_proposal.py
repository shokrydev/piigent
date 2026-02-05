"""Fix Proposal Agent.

Proposes specific fixes based on error analysis,
translating error patterns into prompt mutations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from agents.critics.error_taxonomy import ClassifiedError, ErrorType, ErrorTaxonomyAgent
from prompts.genome import PromptGenotype


class MutationOperator(str, Enum):
    """Mutation operators for prompt changes."""
    # Example mutations
    ADD_EXAMPLE = "add_example"           # Add new few-shot example
    SWAP_EXAMPLE = "swap_example"         # Replace with better example
    REMOVE_EXAMPLE = "remove_example"     # Remove unhelpful example

    # Instruction mutations
    REPHRASE_INSTRUCTION = "rephrase"     # Reword core instruction
    ADD_CONSTRAINT = "add_constraint"     # Add explicit constraint
    LOOSEN_CONSTRAINT = "loosen"          # Remove overly specific constraint

    # Entity-specific
    EMPHASIZE_ENTITY = "emphasize"        # Strengthen entity definition
    ADD_PATTERN = "add_pattern"           # Add pattern to definition
    ADD_CONTEXT_HINT = "add_context"      # Add context keyword hints

    # Disambiguation
    ADD_DISAMBIGUATION = "disambiguate"   # Add type disambiguation


@dataclass
class ProposedFix:
    """A proposed fix for a detection issue."""
    mutation: MutationOperator
    target: str  # What to modify (e.g., "entity_definitions.OCCUPATION")
    rationale: str
    content: Optional[str] = None  # New content to add/replace
    priority: int = 1  # 1=highest priority
    estimated_impact: float = 0.0  # Estimated F1 improvement
    errors_addressed: List[ClassifiedError] = field(default_factory=list)


class FixProposalAgent:
    """Agent that proposes fixes based on error analysis.

    Analyzes error patterns and proposes specific prompt mutations
    to address systematic issues.

    Example:
        taxonomy = ErrorTaxonomyAgent()
        errors = taxonomy.analyze_batch(expected, detected, text)

        proposal_agent = FixProposalAgent()
        fixes = proposal_agent.propose_fixes(errors, current_prompt)

        for fix in fixes:
            print(f"{fix.mutation.value}: {fix.rationale}")
    """

    # Mapping from error types to potential fixes
    ERROR_TO_FIX_MAP = {
        ErrorType.FALSE_NEGATIVE: [
            MutationOperator.ADD_EXAMPLE,
            MutationOperator.EMPHASIZE_ENTITY,
            MutationOperator.ADD_PATTERN,
        ],
        ErrorType.TYPE_CONFUSION: [
            MutationOperator.ADD_DISAMBIGUATION,
            MutationOperator.ADD_CONSTRAINT,
        ],
        ErrorType.BOUNDARY_LEFT: [
            MutationOperator.ADD_EXAMPLE,
            MutationOperator.ADD_CONSTRAINT,
        ],
        ErrorType.BOUNDARY_RIGHT: [
            MutationOperator.ADD_EXAMPLE,
            MutationOperator.ADD_CONSTRAINT,
        ],
        ErrorType.CONTEXT_DEPENDENT: [
            MutationOperator.ADD_CONTEXT_HINT,
            MutationOperator.ADD_EXAMPLE,
        ],
        ErrorType.FORMAT_VARIATION: [
            MutationOperator.ADD_PATTERN,
            MutationOperator.ADD_EXAMPLE,
        ],
        ErrorType.FALSE_POSITIVE: [
            MutationOperator.ADD_CONSTRAINT,
            MutationOperator.LOOSEN_CONSTRAINT,
        ],
    }

    def __init__(
        self,
        min_errors_for_fix: int = 2,  # Minimum errors before proposing fix
    ):
        """Initialize the fix proposal agent.

        Args:
            min_errors_for_fix: Minimum error count to trigger fix
        """
        self.min_errors_for_fix = min_errors_for_fix

    def propose_fixes(
        self,
        errors: List[ClassifiedError],
        current_prompt: PromptGenotype,
    ) -> List[ProposedFix]:
        """Propose fixes based on error analysis.

        Args:
            errors: List of classified errors
            current_prompt: Current prompt genome

        Returns:
            List of proposed fixes, prioritized
        """
        proposals = []

        # Group errors by type and entity type
        by_error_type = self._group_by_error_type(errors)
        by_entity_type = self._group_by_entity_type(errors)

        # Propose fixes for common error patterns
        for error_type, error_group in by_error_type.items():
            if len(error_group) >= self.min_errors_for_fix:
                fix_proposals = self._propose_for_error_type(
                    error_type, error_group, current_prompt
                )
                proposals.extend(fix_proposals)

        # Propose entity-specific fixes
        for entity_type, error_group in by_entity_type.items():
            if len(error_group) >= self.min_errors_for_fix:
                fix_proposals = self._propose_for_entity_type(
                    entity_type, error_group, current_prompt
                )
                proposals.extend(fix_proposals)

        # Sort by priority and estimated impact
        proposals.sort(key=lambda p: (-p.estimated_impact, p.priority))

        # Deduplicate similar proposals
        return self._deduplicate_proposals(proposals)

    def _group_by_error_type(
        self,
        errors: List[ClassifiedError],
    ) -> Dict[ErrorType, List[ClassifiedError]]:
        """Group errors by error type."""
        groups = {}
        for error in errors:
            if error.error_type not in groups:
                groups[error.error_type] = []
            groups[error.error_type].append(error)
        return groups

    def _group_by_entity_type(
        self,
        errors: List[ClassifiedError],
    ) -> Dict[str, List[ClassifiedError]]:
        """Group errors by entity type."""
        groups = {}
        for error in errors:
            if error.expected_type:
                if error.expected_type not in groups:
                    groups[error.expected_type] = []
                groups[error.expected_type].append(error)
        return groups

    def _propose_for_error_type(
        self,
        error_type: ErrorType,
        errors: List[ClassifiedError],
        current_prompt: PromptGenotype,
    ) -> List[ProposedFix]:
        """Propose fixes for a specific error type."""
        proposals = []
        potential_mutations = self.ERROR_TO_FIX_MAP.get(error_type, [])

        for mutation in potential_mutations:
            proposal = self._create_proposal(
                mutation, error_type, errors, current_prompt
            )
            if proposal:
                proposals.append(proposal)

        return proposals

    def _propose_for_entity_type(
        self,
        entity_type: str,
        errors: List[ClassifiedError],
        current_prompt: PromptGenotype,
    ) -> List[ProposedFix]:
        """Propose fixes for a specific entity type."""
        proposals = []

        # Check if entity type has definition
        has_definition = entity_type in current_prompt.entity_definitions

        # Propose adding/improving definition
        if not has_definition or len(errors) >= 3:
            # Get common error patterns
            error_texts = [e.entity_text for e in errors[:5]]
            error_contexts = [e.context for e in errors[:3]]

            proposals.append(ProposedFix(
                mutation=MutationOperator.EMPHASIZE_ENTITY,
                target=f"entity_definitions.{entity_type}",
                rationale=f"High error rate for {entity_type} ({len(errors)} errors)",
                content=self._generate_definition_improvement(
                    entity_type, error_texts, error_contexts,
                    current_prompt.entity_definitions.get(entity_type, "")
                ),
                priority=1,
                estimated_impact=0.05 * len(errors),
                errors_addressed=errors,
            ))

        # Propose adding examples
        miss_errors = [e for e in errors if e.error_type == ErrorType.FALSE_NEGATIVE]
        if len(miss_errors) >= 2:
            proposals.append(ProposedFix(
                mutation=MutationOperator.ADD_EXAMPLE,
                target=f"examples.{entity_type}",
                rationale=f"Many missed {entity_type} entities",
                content=self._generate_example(entity_type, miss_errors[0]),
                priority=2,
                estimated_impact=0.03 * len(miss_errors),
                errors_addressed=miss_errors,
            ))

        return proposals

    def _create_proposal(
        self,
        mutation: MutationOperator,
        error_type: ErrorType,
        errors: List[ClassifiedError],
        current_prompt: PromptGenotype,
    ) -> Optional[ProposedFix]:
        """Create a specific proposal for a mutation."""

        if mutation == MutationOperator.ADD_CONSTRAINT:
            constraint = self._generate_constraint(error_type, errors)
            return ProposedFix(
                mutation=mutation,
                target="constraints",
                rationale=f"Address {error_type.value} errors ({len(errors)} occurrences)",
                content=constraint,
                priority=2,
                estimated_impact=0.02 * len(errors),
                errors_addressed=errors,
            )

        elif mutation == MutationOperator.ADD_DISAMBIGUATION:
            if error_type == ErrorType.TYPE_CONFUSION:
                disambiguation = self._generate_disambiguation(errors)
                if disambiguation:
                    return ProposedFix(
                        mutation=mutation,
                        target="entity_definitions",
                        rationale=f"Disambiguate confused types",
                        content=disambiguation,
                        priority=1,
                        estimated_impact=0.04 * len(errors),
                        errors_addressed=errors,
                    )

        elif mutation == MutationOperator.ADD_CONTEXT_HINT:
            hint = self._generate_context_hint(errors)
            return ProposedFix(
                mutation=mutation,
                target="entity_definitions",
                rationale=f"Add context hints for context-dependent detection",
                content=hint,
                priority=3,
                estimated_impact=0.02 * len(errors),
                errors_addressed=errors,
            )

        return None

    def _generate_constraint(
        self,
        error_type: ErrorType,
        errors: List[ClassifiedError],
    ) -> str:
        """Generate a constraint based on error pattern."""

        if error_type == ErrorType.FALSE_POSITIVE:
            # Find common false positive patterns
            texts = [e.entity_text for e in errors]
            return f"Do NOT tag the following as entities: {', '.join(set(texts[:5]))}"

        elif error_type == ErrorType.BOUNDARY_LEFT:
            return "Entity spans should NOT include preceding titles or prefixes like 'Herr', 'Frau', unless they are part of the entity"

        elif error_type == ErrorType.BOUNDARY_RIGHT:
            return "Entity spans should NOT include trailing punctuation or suffixes"

        elif error_type == ErrorType.TYPE_CONFUSION:
            pairs = [e.metadata.get("confusion_pair") for e in errors if "confusion_pair" in e.metadata]
            if pairs:
                common_pair = max(set(pairs), key=pairs.count)
                return f"Carefully distinguish {common_pair[0]} from {common_pair[1]} - they are different types"

        return "Be precise with entity boundaries and types"

    def _generate_disambiguation(
        self,
        errors: List[ClassifiedError],
    ) -> Optional[str]:
        """Generate disambiguation text for confused types."""
        pairs = [e.metadata.get("confusion_pair") for e in errors if "confusion_pair" in e.metadata]
        if not pairs:
            return None

        common_pair = max(set(pairs), key=pairs.count)
        type_a, type_b = common_pair

        return f"""
Note: {type_a} and {type_b} are different:
- {type_a}: [specific distinguishing features]
- {type_b}: [specific distinguishing features]
"""

    def _generate_context_hint(
        self,
        errors: List[ClassifiedError],
    ) -> str:
        """Generate context hint for detection."""
        contexts = [e.context for e in errors[:5]]
        entity_types = list(set(e.expected_type for e in errors if e.expected_type))

        if not entity_types:
            return ""

        entity_type = entity_types[0]

        # Extract common context patterns
        common_words = self._extract_common_words(contexts)

        return f"Look for {entity_type} near these context words: {', '.join(common_words[:5])}"

    def _generate_definition_improvement(
        self,
        entity_type: str,
        error_texts: List[str],
        contexts: List[str],
        current_definition: str,
    ) -> str:
        """Generate improved entity definition."""
        if current_definition:
            return f"{current_definition}. Examples include: {', '.join(error_texts[:3])}"
        else:
            return f"{entity_type}: Includes items like {', '.join(error_texts[:3])}"

    def _generate_example(
        self,
        entity_type: str,
        error: ClassifiedError,
    ) -> str:
        """Generate a few-shot example from an error."""
        return f"""
Input: "{error.context}"
Entity: "{error.entity_text}" is {entity_type}
"""

    def _extract_common_words(self, texts: List[str]) -> List[str]:
        """Extract common words from text samples."""
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

        return sorted(word_counts.keys(), key=lambda w: -word_counts[w])[:10]

    def _deduplicate_proposals(
        self,
        proposals: List[ProposedFix],
    ) -> List[ProposedFix]:
        """Remove duplicate proposals."""
        seen = set()
        unique = []

        for proposal in proposals:
            key = (proposal.mutation, proposal.target)
            if key not in seen:
                seen.add(key)
                unique.append(proposal)

        return unique

    def apply_fix(
        self,
        fix: ProposedFix,
        genome: PromptGenotype,
    ) -> PromptGenotype:
        """Apply a proposed fix to a genome.

        Args:
            fix: The fix to apply
            genome: The genome to modify

        Returns:
            New genome with fix applied
        """
        new_genome = genome.clone()

        if fix.mutation == MutationOperator.ADD_CONSTRAINT:
            if fix.content:
                new_genome.add_constraint(fix.content)

        elif fix.mutation == MutationOperator.EMPHASIZE_ENTITY:
            if fix.target.startswith("entity_definitions.") and fix.content:
                entity_type = fix.target.split(".")[-1]
                new_genome.update_entity_definition(entity_type, fix.content)

        elif fix.mutation == MutationOperator.ADD_EXAMPLE:
            # Parse target to get entity type
            if fix.content:
                from prompts.genome import PromptExample
                entity_type = fix.target.split(".")[-1] if "." in fix.target else "UNKNOWN"
                example = PromptExample(
                    input_text=fix.content.split("\n")[1] if "\n" in fix.content else "",
                    expected_output="",
                    entity_type=entity_type,
                    source="auto_generated",
                )
                new_genome.add_example(example)

        elif fix.mutation == MutationOperator.ADD_DISAMBIGUATION:
            if fix.content:
                new_genome.add_constraint(fix.content)

        return new_genome
