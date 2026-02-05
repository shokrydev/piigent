"""Weakness to Mutation Mapper.

Maps weakness analysis results to prompt mutations,
enabling automatic prompt evolution based on discovered weaknesses.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agents.critics.error_taxonomy import ClassifiedError, ErrorType
from agents.critics.fix_proposal import MutationOperator, ProposedFix
from agents.core.weakness_analyzer import WeaknessType


# =============================================================================
# Weakness to Mutation Mapping
# =============================================================================

# Maps WeaknessType to appropriate MutationOperators
WEAKNESS_MUTATION_MAP: Dict[str, List[MutationOperator]] = {
    WeaknessType.OVERLAP_CONFLICT.value: [
        MutationOperator.ADD_CONSTRAINT,      # Add constraint to prefer pattern recognizers
        MutationOperator.EMPHASIZE_ENTITY,    # Emphasize the specific entity type
        MutationOperator.ADD_DISAMBIGUATION,  # Add disambiguation for overlapping types
    ],
    WeaknessType.SCORE_THRESHOLD.value: [
        MutationOperator.ADD_EXAMPLE,         # Add examples to improve confidence
        MutationOperator.EMPHASIZE_ENTITY,    # Emphasize entity for higher scores
    ],
    WeaknessType.CONTEXT_DEPENDENCY.value: [
        MutationOperator.ADD_CONTEXT_HINT,    # Add context hints for detection
        MutationOperator.ADD_EXAMPLE,         # Add examples with varied context
        MutationOperator.ADD_PATTERN,         # Add pattern for context triggers
    ],
    WeaknessType.FORMAT_VARIATION.value: [
        MutationOperator.ADD_PATTERN,         # Add patterns for variations
        MutationOperator.ADD_EXAMPLE,         # Add examples of varied formats
    ],
    WeaknessType.ENTITY_CONFUSION.value: [
        MutationOperator.ADD_DISAMBIGUATION,  # Disambiguate confused types
        MutationOperator.ADD_CONSTRAINT,      # Add constraints to clarify
        MutationOperator.ADD_EXAMPLE,         # Add examples of each type
    ],
    WeaknessType.COVERAGE_GAP.value: [
        MutationOperator.ADD_EXAMPLE,         # Add examples for uncovered entity
        MutationOperator.ADD_PATTERN,         # Add detection pattern
        MutationOperator.REPHRASE_INSTRUCTION,  # Rephrase to include entity
    ],
    WeaknessType.LEAKAGE.value: [
        MutationOperator.ADD_CONSTRAINT,      # Add constraints for thorough detection
        MutationOperator.ADD_EXAMPLE,         # Add examples of edge cases
    ],
}


class WeaknessToMutationMapper:
    """Maps weakness analysis results to prompt mutations.
    
    Converts weaknesses discovered by the WeaknessAnalyzer into
    ProposedFix objects that can be applied via the MutationEngine.
    
    Example:
        mapper = WeaknessToMutationMapper()
        fixes = mapper.map_weakness_to_mutations(weakness)
        for fix in fixes:
            genome = mutation_engine.apply_mutation(
                genome, 
                fix.mutation, 
                {"content": fix.content, "target": fix.target}
            )
    """
    
    def __init__(self, max_mutations_per_weakness: int = 3):
        """Initialize the mapper.
        
        Args:
            max_mutations_per_weakness: Maximum mutations to propose per weakness
        """
        self.max_mutations_per_weakness = max_mutations_per_weakness
    
    def map_weakness_to_mutations(
        self,
        weakness: Dict,
        current_prompt_info: Optional[Dict] = None,
    ) -> List[ProposedFix]:
        """Convert a weakness to mutation proposals.
        
        Args:
            weakness: Weakness dict from WeaknessAnalyzer
            current_prompt_info: Optional info about current prompt state
        
        Returns:
            List of ProposedFix objects for the weakness
        """
        weakness_type = weakness.get("weakness_type", "")
        entity_type = weakness.get("entity_type", "")
        severity = weakness.get("severity", 0.5)
        evidence = weakness.get("evidence", [])
        root_cause = weakness.get("root_cause", "")
        suggested_fix = weakness.get("suggested_fix", "")
        
        # Get appropriate mutation operators for this weakness type
        operators = WEAKNESS_MUTATION_MAP.get(
            weakness_type, 
            [MutationOperator.ADD_EXAMPLE]  # Default fallback
        )
        
        proposals = []
        
        for i, operator in enumerate(operators[:self.max_mutations_per_weakness]):
            # Generate content based on evidence and weakness type
            content = self._generate_mutation_content(
                operator=operator,
                entity_type=entity_type,
                weakness_type=weakness_type,
                evidence=evidence,
                root_cause=root_cause,
                suggested_fix=suggested_fix,
            )
            
            # Calculate priority based on severity and operator order
            priority = max(1, int(3 - severity * 2)) + i
            
            proposal = ProposedFix(
                mutation=operator,
                target=entity_type,
                rationale=f"Address {weakness_type} weakness for {entity_type}: {root_cause[:100] if root_cause else 'unconfirmed'}",
                content=content,
                priority=priority,
                estimated_impact=severity * (1.0 - i * 0.2),  # Diminishing returns
                errors_addressed=[],  # Will be populated if we convert evidence to errors
            )
            
            proposals.append(proposal)
        
        return proposals
    
    def _generate_mutation_content(
        self,
        operator: MutationOperator,
        entity_type: str,
        weakness_type: str,
        evidence: List[Dict],
        root_cause: str,
        suggested_fix: str,
    ) -> Optional[str]:
        """Generate content for a mutation based on weakness context.
        
        Args:
            operator: The mutation operator to generate content for
            entity_type: The entity type being addressed
            weakness_type: The type of weakness
            evidence: Evidence supporting the weakness
            root_cause: Identified root cause
            suggested_fix: Suggested fix from analysis
        
        Returns:
            Content string for the mutation, or None if not applicable
        """
        if operator == MutationOperator.ADD_EXAMPLE:
            # Generate example from evidence
            return self._generate_example_from_evidence(entity_type, evidence)
        
        elif operator == MutationOperator.ADD_CONSTRAINT:
            # Generate constraint based on weakness
            return self._generate_constraint(entity_type, weakness_type, root_cause)
        
        elif operator == MutationOperator.ADD_DISAMBIGUATION:
            # Generate disambiguation text
            return self._generate_disambiguation(entity_type, weakness_type, evidence)
        
        elif operator == MutationOperator.ADD_CONTEXT_HINT:
            # Generate context hint
            return self._generate_context_hint(entity_type, evidence)
        
        elif operator == MutationOperator.ADD_PATTERN:
            # Generate pattern from evidence
            return self._generate_pattern(entity_type, evidence)
        
        elif operator == MutationOperator.EMPHASIZE_ENTITY:
            # Generate emphasis text
            return f"Pay special attention to {entity_type} entities."
        
        elif operator == MutationOperator.REPHRASE_INSTRUCTION:
            # Generate rephrase content suggesting entity inclusion
            return f"Ensure {entity_type} entities are detected in all contexts."
        
        return None
    
    def _generate_example_from_evidence(
        self,
        entity_type: str,
        evidence: List[Dict],
    ) -> Optional[str]:
        """Generate an example from weakness evidence."""
        if not evidence:
            return None
        
        # Use first evidence item to create example
        first = evidence[0]
        test_input = first.get("test_input", "")
        expected = first.get("expected", {})
        
        if test_input and expected:
            expected_values = expected.get(entity_type, [])
            if expected_values:
                return f"Input: {test_input[:100]}\nOutput: {entity_type}: {expected_values[0]}"
        
        return None
    
    def _generate_constraint(
        self,
        entity_type: str,
        weakness_type: str,
        root_cause: str,
    ) -> str:
        """Generate a constraint based on weakness type."""
        if weakness_type == WeaknessType.OVERLAP_CONFLICT.value:
            return f"When detecting {entity_type}, prefer exact pattern matches over broader entity types."
        elif weakness_type == WeaknessType.LEAKAGE.value:
            return f"Ensure all variations of {entity_type} are detected, including partial mentions."
        elif weakness_type == WeaknessType.ENTITY_CONFUSION.value:
            return f"Carefully distinguish {entity_type} from similar entity types based on context and format."
        else:
            return f"Apply strict detection criteria for {entity_type}."
    
    def _generate_disambiguation(
        self,
        entity_type: str,
        weakness_type: str,
        evidence: List[Dict],
    ) -> str:
        """Generate disambiguation text."""
        if weakness_type == WeaknessType.ENTITY_CONFUSION.value:
            return (
                f"{entity_type} should be distinguished from similar types by examining "
                f"the surrounding context and specific format patterns."
            )
        elif weakness_type == WeaknessType.OVERLAP_CONFLICT.value:
            return (
                f"When {entity_type} appears within a larger entity (like a location), "
                f"extract both the component {entity_type} and the larger entity."
            )
        else:
            return f"Ensure {entity_type} is correctly identified in all contexts."
    
    def _generate_context_hint(
        self,
        entity_type: str,
        evidence: List[Dict],
    ) -> str:
        """Generate context hint from evidence."""
        # Analyze evidence for context patterns
        contexts = []
        for e in evidence[:3]:
            test_input = e.get("test_input", "")
            if test_input:
                # Extract surrounding words
                words = test_input.split()[:5]
                if words:
                    contexts.append(" ".join(words))
        
        if contexts:
            return f"{entity_type} often appears in contexts like: {'; '.join(contexts)}"
        return f"Look for {entity_type} in relevant contexts."
    
    def _generate_pattern(
        self,
        entity_type: str,
        evidence: List[Dict],
    ) -> Optional[str]:
        """Generate pattern from evidence."""
        if not evidence:
            return None
        
        # Extract example values from evidence
        examples = []
        for e in evidence[:3]:
            expected = e.get("expected", {})
            values = expected.get(entity_type, [])
            examples.extend(values[:2])
        
        if examples:
            return f"Pattern examples for {entity_type}: {', '.join(examples[:5])}"
        return None
    
    def convert_evidence_to_errors(
        self,
        evidence: List[Dict],
        entity_type: str,
    ) -> List[ClassifiedError]:
        """Convert weakness evidence to ClassifiedError format.
        
        Enables use of FixProposalAgent for error-based mutation proposals.
        
        Args:
            evidence: Evidence from weakness analysis
            entity_type: The entity type involved
        
        Returns:
            List of ClassifiedError objects
        """
        errors = []
        
        for e in evidence:
            test_input = e.get("test_input", "")
            expected = e.get("expected", {})
            actual = e.get("actual", {})
            explanation = e.get("explanation", "")
            
            expected_values = expected.get(entity_type, [])
            detected = actual.get("detected", [])
            
            # Determine error type based on detection result
            entity_detected = any(
                d.get("entity_type") == entity_type 
                for d in detected
            )
            
            if not entity_detected and expected_values:
                # False negative
                error_type = ErrorType.FALSE_NEGATIVE
                detected_type = None
                detected_span = None
            else:
                # Other errors (type confusion, boundary, etc.)
                error_type = ErrorType.TYPE_CONFUSION
                detected_type = detected[0].get("entity_type") if detected else None
                detected_span = (
                    (detected[0].get("start", 0), detected[0].get("end", 0))
                    if detected else None
                )
            
            error = ClassifiedError(
                error_type=error_type,
                entity_text=expected_values[0] if expected_values else "",
                expected_type=entity_type,
                detected_type=detected_type,
                expected_span=(0, len(expected_values[0]) if expected_values else 0),
                detected_span=detected_span,
                context=test_input[:200],
                explanation=explanation or f"Expected {entity_type} not detected correctly",
                severity=1.0,
                metadata={"source": "weakness_evidence"},
            )
            
            errors.append(error)
        
        return errors
    
    def map_weaknesses_batch(
        self,
        weaknesses: List[Dict],
        max_total_mutations: int = 10,
    ) -> List[ProposedFix]:
        """Map multiple weaknesses to mutations with prioritization.
        
        Args:
            weaknesses: List of weaknesses from WeaknessAnalyzer
            max_total_mutations: Maximum total mutations to return
        
        Returns:
            Prioritized list of ProposedFix objects
        """
        all_proposals = []
        
        for weakness in weaknesses:
            proposals = self.map_weakness_to_mutations(weakness)
            all_proposals.extend(proposals)
        
        # Sort by priority (lower is higher priority)
        all_proposals.sort(key=lambda p: (p.priority, -p.estimated_impact))
        
        return all_proposals[:max_total_mutations]
