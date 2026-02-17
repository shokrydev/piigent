"""Blue Team Optimizer - Defensive Mutation Engine.

Translates Red Team findings (vulnerabilities) into defensive
mutations (patches) for the prompt genome.
"""

from typing import Dict, List, Optional
import logging

from agents.critics.error_taxonomy import ClassifiedError, ErrorType
from agents.critics.fix_proposal import MutationOperator, ProposedFix
from synpii.adversarial.types import AdversarialType

logger = logging.getLogger(__name__)


# =============================================================================
# Vulnerability to Patch Mapping
# =============================================================================

# Maps AdversarialType to appropriate MutationOperators
FAILURE_MODE_MUTATION_MAP: Dict[str, List[MutationOperator]] = {
    AdversarialType.OVERLAP_CONFLICT.value: [
        MutationOperator.ADD_CONSTRAINT,      # Add constraint to prefer pattern recognizers
        MutationOperator.EMPHASIZE_ENTITY,    # Emphasize the specific entity type
        MutationOperator.ADD_DISAMBIGUATION,  # Add disambiguation for overlapping types
    ],
    AdversarialType.CONTEXT_DEPENDENCY.value: [
        MutationOperator.ADD_CONTEXT_HINT,    # Add context hints for detection
        MutationOperator.ADD_EXAMPLE,         # Add examples with varied context
        MutationOperator.ADD_PATTERN,         # Add pattern for context triggers
    ],
    AdversarialType.FORMAT_VARIATION.value: [
        MutationOperator.ADD_PATTERN,         # Add patterns for variations
        MutationOperator.ADD_EXAMPLE,         # Add examples of varied formats
    ],
    AdversarialType.ENTITY_CONFUSION.value: [
        MutationOperator.ADD_DISAMBIGUATION,  # Disambiguate confused types
        MutationOperator.ADD_CONSTRAINT,      # Add constraints to clarify
        MutationOperator.ADD_EXAMPLE,         # Add examples of each type
    ],
    AdversarialType.COVERAGE_GAP.value: [
        MutationOperator.ADD_EXAMPLE,         # Add examples for uncovered entity
        MutationOperator.ADD_PATTERN,         # Add detection pattern
        MutationOperator.REPHRASE_INSTRUCTION,  # Rephrase to include entity
    ],
    AdversarialType.LEAKAGE.value: [
        MutationOperator.ADD_CONSTRAINT,      # Add constraints for thorough detection
        MutationOperator.ADD_EXAMPLE,         # Add examples of edge cases
    ],
}


class BlueTeamOptimizer:
    """Blue Team Optimization Engine.

    Translates Red Team findings (vulnerabilities) into defensive
    mutations (patches) for the prompt genome.

    Example:
        optimizer = BlueTeamOptimizer()
        patches = optimizer.map_finding_to_patches(finding)
    """

    def __init__(self, max_mutations_per_weakness: int = 3):
        """Initialize the optimizer.

        Args:
            max_mutations_per_weakness: Maximum mutations to propose per finding
        """
        self.max_mutations_per_weakness = max_mutations_per_weakness

    def map_weaknesses_batch(
        self,
        findings: List[Dict],
        max_total_mutations: int = 10,
    ) -> List[ProposedFix]:
        """Map multiple findings to mutations with prioritization.

        Args:
            findings: List of AdversarialFindings
            max_total_mutations: Maximum total mutations to return

        Returns:
            Prioritized list of ProposedFix objects
        """
        all_proposals = []

        for finding in findings:
            proposals = self.map_finding_to_patches(finding)
            all_proposals.extend(proposals)

        # Sort by priority (lower is higher priority) and impact
        # We want high impact first, but usually priority logic handles ordering.
        # Let's trust priority logic: lower priority number = more urgent.
        all_proposals.sort(key=lambda p: (p.priority, -p.estimated_impact))

        return all_proposals[:max_total_mutations]

    def map_finding_to_patches(
        self,
        finding: Dict,
        current_prompt_info: Optional[Dict] = None,
    ) -> List[ProposedFix]:
        """Convert a finding to mutation proposals.

        Args:
            finding: AdversarialFinding dict from Orchestrator
            current_prompt_info: Optional info about current prompt state

        Returns:
            List of ProposedFix objects (Patches)
        """
        failure_mode = finding.get("failure_mode", "")
        # Fallback for old dicts if any
        if not failure_mode:
            failure_mode = finding.get("weakness_type", "")
            
        entity_type = finding.get("entity_type", "")
        severity = finding.get("severity", 0.5)
        evidence = finding.get("evidence", [])
        root_cause = finding.get("root_cause", "")
        suggested_fix = finding.get("defense_strategy", finding.get("suggested_fix", ""))

        # Get appropriate mutation operators
        operators = FAILURE_MODE_MUTATION_MAP.get(
            failure_mode,
            [MutationOperator.ADD_EXAMPLE]  # Default fallback
        )

        proposals = []

        for i, operator in enumerate(operators[:self.max_mutations_per_weakness]):
            # Generate content
            content = self._generate_patch_content(
                operator=operator,
                entity_type=entity_type,
                failure_mode=failure_mode,
                evidence=evidence,
                root_cause=root_cause,
                suggested_fix=suggested_fix,
            )

            # Calculate priority
            # Higher severity = Lower priority index (1 is highest)
            priority = max(1, int(3 - severity * 2)) + i

            proposal = ProposedFix(
                mutation=operator,
                target=entity_type,
                rationale=f"Fix {failure_mode} for {entity_type}: {root_cause[:100]}",
                content=content,
                priority=priority,
                estimated_impact=severity * (1.0 - i * 0.2),
                errors_addressed=[],
            )

            proposals.append(proposal)

        return proposals

    def _generate_patch_content(
        self,
        operator: MutationOperator,
        entity_type: str,
        failure_mode: str,
        evidence: List[Dict],
        root_cause: str,
        suggested_fix: str,
    ) -> Optional[str]:
        """Generate content for a mutation patch."""
        if operator == MutationOperator.ADD_EXAMPLE:
            return self._generate_example(entity_type, evidence)

        elif operator == MutationOperator.ADD_CONSTRAINT:
            return self._generate_constraint(entity_type, failure_mode)

        elif operator == MutationOperator.ADD_DISAMBIGUATION:
            return self._generate_disambiguation(entity_type, failure_mode)

        elif operator == MutationOperator.ADD_CONTEXT_HINT:
            return self._generate_context_hint(entity_type, evidence)

        elif operator == MutationOperator.ADD_PATTERN:
            return self._generate_pattern(entity_type, evidence)

        elif operator == MutationOperator.EMPHASIZE_ENTITY:
            return f"Pay special attention to {entity_type} entities."

        elif operator == MutationOperator.REPHRASE_INSTRUCTION:
            return f"Ensure {entity_type} entities are detected in all contexts."

        return None

    def _generate_example(self, entity_type: str, evidence: List[Dict]) -> Optional[str]:
        """Generate an example from evidence."""
        if not evidence:
            return None
        
        # Evidence struct: {test_input, expected, ...}
        first = evidence[0]
        test_input = first.get("test_input", "")
        expected = first.get("expected", {})
        
        if test_input and expected:
            vals = expected.get(entity_type, [])
            if vals:
                 return f"Input: {test_input[:150]}\nOutput: {entity_type}: {vals[0]}"
        return None

    def _generate_constraint(self, entity_type: str, failure_mode: str) -> str:
        """Generate a constraint based on failure mode."""
        if failure_mode == AdversarialType.OVERLAP_CONFLICT.value:
            return f"When detecting {entity_type}, prefer exact pattern matches over broader entity types."
        elif failure_mode == AdversarialType.LEAKAGE.value:
            return f"Ensure all variations of {entity_type} are detected, including partial mentions."
        elif failure_mode == AdversarialType.ENTITY_CONFUSION.value:
            return f"Carefully distinguish {entity_type} from similar entity types based on context."
        else:
            return f"Apply strict detection criteria for {entity_type}."

    def _generate_disambiguation(self, entity_type: str, failure_mode: str) -> str:
        """Generate disambiguation text."""
        if failure_mode == AdversarialType.ENTITY_CONFUSION.value:
            return f"{entity_type} should be distinguished from similar types by examining context."
        elif failure_mode == AdversarialType.OVERLAP_CONFLICT.value:
            return f"Extract {entity_type} even if it appears within a larger entity."
        return f"Ensure {entity_type} is correctly identified."

    def _generate_context_hint(self, entity_type: str, evidence: List[Dict]) -> str:
        """Generate context hint."""
        # Simple extraction of surrounding words could be added here
        return f"Look for {entity_type} in relevant contexts."

    def _generate_pattern(self, entity_type: str, evidence: List[Dict]) -> Optional[str]:
        """Generate pattern hint."""
        if not evidence:
            return None
        # Extract example values
        vals = []
        for e in evidence[:3]:
            curr_vals = e.get("expected", {}).get(entity_type, [])
            if curr_vals:
                vals.extend(curr_vals)
        
        if vals:
            return f"Pattern examples for {entity_type}: {', '.join(vals[:3])}"
        return None

    def convert_evidence_to_errors(
        self,
        evidence: List[Dict],
        entity_type: str,
    ) -> List[ClassifiedError]:
        """Convert evidence to ClassifiedError format for Critics."""
        errors = []
        for e in evidence:
            test_input = e.get("test_input", "")
            expected = e.get("expected", {})
            actual = e.get("actual", {})
            explanation = e.get("explanation", "")
            
            expected_vals = expected.get(entity_type, [])
            detected = actual.get("detected", [])
            
            # Simple error logic
            found = False
            for d in detected:
                if d.get("entity_type") == entity_type:
                    found = True
                    break
            
            if not found and expected_vals:
                error = ClassifiedError(
                    error_type=ErrorType.FALSE_NEGATIVE,
                    entity_text=expected_vals[0],
                    expected_type=entity_type,
                    detected_type=None,
                    expected_span=(0, 0),
                    detected_span=None,
                    context=test_input[:100],
                    explanation=explanation or "Not detected",
                    severity=1.0,
                    metadata={"source": "red_team_evidence"}
                )
                errors.append(error)
                
        return errors
