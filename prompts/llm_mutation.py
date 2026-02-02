"""LLM-Guided Mutation.

Uses an LLM to analyze failures and propose intelligent mutations
based on error patterns, rather than random mutations.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prompts.genome import PromptGenotype
from prompts.mutations import MutationOperator

logger = logging.getLogger(__name__)


@dataclass
class MutationProposal:
    """An LLM-proposed mutation."""
    operator: MutationOperator
    target: str
    content: str
    rationale: str
    confidence: float
    error_addressed: str


class LLMMutationAdvisor:
    """Uses LLM to propose intelligent mutations.

    Analyzes failure patterns and uses LLM reasoning to propose
    specific, targeted mutations rather than random changes.

    Example:
        advisor = LLMMutationAdvisor()

        proposals = advisor.propose_mutations(
            genome=current_genome,
            failure_analysis={
                "missed_entities": [...],
                "type_confusions": [...],
            },
        )

        for proposal in proposals:
            print(f"{proposal.operator}: {proposal.rationale}")
    """

    ANALYSIS_PROMPT = """You are analyzing a PII detection prompt that has been failing on certain cases.

Current prompt instruction block:
{instruction_block}

Current entity definitions:
{entity_definitions}

Current constraints:
{constraints}

FAILURE ANALYSIS:
{failure_analysis}

Based on this failure analysis, propose specific improvements to the prompt.
For each improvement, specify:
1. What to change (instruction, definition, constraint, or example)
2. The specific change to make
3. Why this will help

Return your proposals as a JSON array with objects containing:
- "target": what to change ("instruction", "definition:ENTITY_TYPE", "constraint", "example")
- "change": the specific text/content to add or modify
- "rationale": why this helps
- "confidence": 0-1 how confident you are this will help

Return ONLY the JSON array."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        timeout: float = 60.0,
    ):
        """Initialize the LLM mutation advisor.

        Args:
            ollama_url: Ollama server URL
            model: Model to use
            timeout: Request timeout
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Get HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.ollama_url,
                    timeout=self.timeout,
                )
            except ImportError:
                logger.warning("httpx not available")
                return None
        return self._client

    def propose_mutations(
        self,
        genome: PromptGenotype,
        failure_analysis: Dict,
        max_proposals: int = 5,
    ) -> List[MutationProposal]:
        """Propose mutations based on failure analysis.

        Args:
            genome: Current prompt genome
            failure_analysis: Dict with failure information
            max_proposals: Maximum number of proposals

        Returns:
            List of mutation proposals
        """
        client = self._get_client()
        if client is None:
            return self._fallback_proposals(genome, failure_analysis)

        # Build prompt
        prompt = self.ANALYSIS_PROMPT.format(
            instruction_block=genome.instruction_block[:500],
            entity_definitions=json.dumps(genome.entity_definitions, indent=2),
            constraints="\n".join(f"- {c}" for c in genome.constraints[:5]),
            failure_analysis=self._format_failure_analysis(failure_analysis),
        )

        try:
            response = client.post("/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5},
                "format": "json",
            })
            response.raise_for_status()
            result = response.json().get("response", "")

            proposals = self._parse_proposals(result)
            return proposals[:max_proposals]

        except Exception as e:
            logger.error(f"LLM mutation proposal failed: {e}")
            return self._fallback_proposals(genome, failure_analysis)

    def _format_failure_analysis(self, analysis: Dict) -> str:
        """Format failure analysis for the prompt."""
        parts = []

        if "missed_entities" in analysis:
            parts.append("Missed entities:")
            for entity in analysis["missed_entities"][:5]:
                parts.append(f"  - '{entity.get('text')}' as {entity.get('type')}")

        if "type_confusions" in analysis:
            parts.append("\nType confusions:")
            for confusion in analysis["type_confusions"][:5]:
                parts.append(
                    f"  - Detected {confusion.get('detected_type')} "
                    f"instead of {confusion.get('expected_type')}"
                )

        if "boundary_errors" in analysis:
            parts.append(f"\nBoundary errors: {analysis['boundary_errors']} cases")

        if "by_entity_type" in analysis:
            parts.append("\nError counts by entity type:")
            for entity_type, count in sorted(
                analysis["by_entity_type"].items(),
                key=lambda x: -x[1]
            )[:5]:
                parts.append(f"  - {entity_type}: {count} errors")

        return "\n".join(parts) if parts else "No specific failure patterns identified."

    def _parse_proposals(self, response: str) -> List[MutationProposal]:
        """Parse LLM response into mutation proposals."""
        proposals = []

        try:
            data = json.loads(response)
            if isinstance(data, dict) and "proposals" in data:
                data = data["proposals"]
            if not isinstance(data, list):
                data = [data]

            for item in data:
                target = item.get("target", "")
                change = item.get("change", "")
                rationale = item.get("rationale", "")
                confidence = float(item.get("confidence", 0.5))

                operator = self._target_to_operator(target)

                proposals.append(MutationProposal(
                    operator=operator,
                    target=target,
                    content=change,
                    rationale=rationale,
                    confidence=confidence,
                    error_addressed=item.get("error_addressed", ""),
                ))

        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response: {response[:200]}")

        return proposals

    def _target_to_operator(self, target: str) -> MutationOperator:
        """Convert target string to mutation operator."""
        target_lower = target.lower()

        if target_lower.startswith("definition"):
            return MutationOperator.UPDATE_DEFINITION
        elif target_lower == "instruction":
            return MutationOperator.REPHRASE_INSTRUCTION
        elif target_lower == "constraint":
            return MutationOperator.ADD_CONSTRAINT
        elif target_lower == "example":
            return MutationOperator.ADD_EXAMPLE
        else:
            return MutationOperator.ADD_CONSTRAINT

    def _fallback_proposals(
        self,
        genome: PromptGenotype,
        failure_analysis: Dict,
    ) -> List[MutationProposal]:
        """Generate fallback proposals when LLM is unavailable."""
        proposals = []

        # Propose based on error patterns
        if failure_analysis.get("by_entity_type"):
            worst_type = max(
                failure_analysis["by_entity_type"].items(),
                key=lambda x: x[1]
            )[0]

            proposals.append(MutationProposal(
                operator=MutationOperator.UPDATE_DEFINITION,
                target=f"definition:{worst_type}",
                content=f"Improve detection of {worst_type} entities",
                rationale=f"High error rate for {worst_type}",
                confidence=0.5,
                error_addressed=worst_type,
            ))

        if failure_analysis.get("type_confusions"):
            proposals.append(MutationProposal(
                operator=MutationOperator.ADD_CONSTRAINT,
                target="constraint",
                content="Carefully distinguish between similar entity types",
                rationale="Type confusion errors detected",
                confidence=0.4,
                error_addressed="type_confusion",
            ))

        return proposals

    def refine_proposal(
        self,
        proposal: MutationProposal,
        genome: PromptGenotype,
    ) -> MutationProposal:
        """Refine a proposal with more specific content.

        Args:
            proposal: Initial proposal
            genome: Current genome

        Returns:
            Refined proposal with specific content
        """
        client = self._get_client()
        if client is None:
            return proposal

        prompt = f"""Given this mutation proposal for a PII detection prompt:

Operator: {proposal.operator.value}
Target: {proposal.target}
Initial suggestion: {proposal.content}
Rationale: {proposal.rationale}

Current definition (if applicable):
{genome.entity_definitions.get(proposal.target.split(':')[-1], 'N/A')}

Write the EXACT text that should be added/modified. Be specific and actionable.
Return ONLY the new text content, nothing else."""

        try:
            response = client.post("/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            })
            response.raise_for_status()
            refined_content = response.json().get("response", "").strip()

            if refined_content:
                proposal.content = refined_content

        except Exception as e:
            logger.error(f"Proposal refinement failed: {e}")

        return proposal
