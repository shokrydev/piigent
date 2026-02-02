"""Leakage Detection.

Detects when prompts or models memorize synthetic data patterns
rather than learning generalizable detection.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from prompts.genome import PromptGenotype


@dataclass
class LeakageReport:
    """Report on leakage detection."""
    leakage_detected: bool
    leakage_type: str  # "template", "pattern", "example", "none"
    leaked_items: List[str]
    severity: float  # 0-1
    recommendation: str
    details: Dict = field(default_factory=dict)


class LeakageChecker:
    """Checks for leakage of synthetic patterns into prompts.

    Detects when:
    - Prompts contain SynPII-specific templates
    - Examples are too similar to synthetic data
    - Patterns only work on synthetic formats

    Example:
        checker = LeakageChecker(synpii_templates)

        report = checker.check_leakage(prompt)
        if report.leakage_detected:
            print(f"Leakage found: {report.leaked_items}")
    """

    def __init__(
        self,
        synpii_templates: Optional[List[str]] = None,
        synpii_patterns: Optional[List[str]] = None,
    ):
        """Initialize the leakage checker.

        Args:
            synpii_templates: SynPII template strings to check against
            synpii_patterns: SynPII-specific patterns to detect
        """
        self.synpii_templates = synpii_templates or []
        self.synpii_patterns = synpii_patterns or self._default_patterns()

    def _default_patterns(self) -> List[str]:
        """Default SynPII-specific patterns to check."""
        return [
            # Template markers
            r"\{[A-Z_]+\}",  # {ENTITY_TYPE} placeholders
            r"<[A-Z_]+>",    # <ENTITY_TYPE> markers

            # Synthetic name patterns (overly regular)
            r"Person_\d+",
            r"Location_\d+",
            r"Org_\d+",

            # Test data markers
            r"TEST_",
            r"SAMPLE_",
            r"SYNTHETIC_",

            # Numbered sequences (synthetic artifact)
            r"Patient_\d{3}",
            r"Case_\d{4}",
        ]

    def check_leakage(
        self,
        prompt: PromptGenotype,
    ) -> LeakageReport:
        """Check a prompt for synthetic data leakage.

        Args:
            prompt: The prompt genome to check

        Returns:
            LeakageReport with findings
        """
        leaked_items = []
        leakage_type = "none"
        severity = 0.0

        # Check instruction block
        instruction_leaks = self._check_text(prompt.instruction_block)
        if instruction_leaks:
            leaked_items.extend(instruction_leaks)
            leakage_type = "template"
            severity = max(severity, 0.8)

        # Check examples
        example_leaks = []
        for example in prompt.examples:
            leaks = self._check_text(example.input_text)
            leaks.extend(self._check_text(example.expected_output))
            example_leaks.extend(leaks)

        if example_leaks:
            leaked_items.extend(example_leaks)
            if leakage_type == "none":
                leakage_type = "example"
            severity = max(severity, 0.6)

        # Check entity definitions
        definition_leaks = []
        for entity_type, definition in prompt.entity_definitions.items():
            leaks = self._check_text(definition)
            definition_leaks.extend(leaks)

        if definition_leaks:
            leaked_items.extend(definition_leaks)
            if leakage_type == "none":
                leakage_type = "pattern"
            severity = max(severity, 0.5)

        # Check constraints
        constraint_leaks = []
        for constraint in prompt.constraints:
            leaks = self._check_text(constraint)
            constraint_leaks.extend(leaks)

        if constraint_leaks:
            leaked_items.extend(constraint_leaks)
            severity = max(severity, 0.3)

        # Generate recommendation
        if severity >= 0.8:
            recommendation = "remove_templates"
        elif severity >= 0.5:
            recommendation = "diversify_examples"
        elif severity >= 0.3:
            recommendation = "review_patterns"
        else:
            recommendation = "none"

        return LeakageReport(
            leakage_detected=len(leaked_items) > 0,
            leakage_type=leakage_type,
            leaked_items=list(set(leaked_items)),
            severity=severity,
            recommendation=recommendation,
            details={
                "instruction_leaks": instruction_leaks,
                "example_leaks": example_leaks,
                "definition_leaks": definition_leaks,
                "constraint_leaks": constraint_leaks,
            },
        )

    def _check_text(self, text: str) -> List[str]:
        """Check text for leakage patterns."""
        found = []

        # Check against templates
        for template in self.synpii_templates:
            if template.lower() in text.lower():
                found.append(f"template:{template[:50]}")

        # Check against patterns
        for pattern in self.synpii_patterns:
            try:
                matches = re.findall(pattern, text)
                found.extend(f"pattern:{m}" for m in matches)
            except re.error:
                pass

        return found

    def check_example_similarity(
        self,
        prompt_examples: List[str],
        synthetic_examples: List[str],
        similarity_threshold: float = 0.8,
    ) -> List[Tuple[str, str, float]]:
        """Check if prompt examples are too similar to synthetic data.

        Args:
            prompt_examples: Examples in the prompt
            synthetic_examples: Known synthetic examples
            similarity_threshold: Similarity threshold to flag

        Returns:
            List of (prompt_example, synth_example, similarity) tuples
        """
        similar_pairs = []

        for prompt_ex in prompt_examples:
            for synth_ex in synthetic_examples:
                sim = self._text_similarity(prompt_ex, synth_ex)
                if sim >= similarity_threshold:
                    similar_pairs.append((prompt_ex, synth_ex, sim))

        return similar_pairs

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard on words)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def check_memorization(
        self,
        pipeline,
        synthetic_samples: List[Dict],
        novel_samples: List[Dict],
    ) -> Dict:
        """Check if pipeline memorized synthetic data vs generalizing.

        Args:
            pipeline: Detection pipeline
            synthetic_samples: Samples similar to training
            novel_samples: Novel samples different from training

        Returns:
            Memorization analysis
        """
        # Evaluate on both
        synth_correct = 0
        synth_total = 0
        novel_correct = 0
        novel_total = 0

        for sample in synthetic_samples:
            text = sample.get("text", "")
            expected = sample.get("entities", [])

            try:
                detected = pipeline(text)
                if not isinstance(detected, list):
                    detected = []

                # Count exact matches
                expected_spans = {(e["start"], e["end"]) for e in expected}
                detected_spans = {(d.get("start", 0), d.get("end", 0)) for d in detected}

                synth_correct += len(expected_spans & detected_spans)
                synth_total += len(expected_spans)
            except Exception:
                synth_total += len(expected)

        for sample in novel_samples:
            text = sample.get("text", "")
            expected = sample.get("entities", [])

            try:
                detected = pipeline(text)
                if not isinstance(detected, list):
                    detected = []

                expected_spans = {(e["start"], e["end"]) for e in expected}
                detected_spans = {(d.get("start", 0), d.get("end", 0)) for d in detected}

                novel_correct += len(expected_spans & detected_spans)
                novel_total += len(expected_spans)
            except Exception:
                novel_total += len(expected)

        synth_accuracy = synth_correct / synth_total if synth_total > 0 else 0
        novel_accuracy = novel_correct / novel_total if novel_total > 0 else 0

        # Large gap suggests memorization
        gap = synth_accuracy - novel_accuracy
        is_memorizing = gap > 0.2  # 20% gap

        return {
            "synthetic_accuracy": synth_accuracy,
            "novel_accuracy": novel_accuracy,
            "gap": gap,
            "is_memorizing": is_memorizing,
            "recommendation": "diversify_training" if is_memorizing else "continue",
        }

    def sanitize_prompt(
        self,
        prompt: PromptGenotype,
    ) -> PromptGenotype:
        """Remove leaked patterns from a prompt.

        Args:
            prompt: Prompt to sanitize

        Returns:
            Sanitized prompt
        """
        sanitized = prompt.clone()

        # Remove leaked patterns from instruction
        sanitized.instruction_block = self._sanitize_text(sanitized.instruction_block)

        # Sanitize entity definitions
        sanitized.entity_definitions = {
            k: self._sanitize_text(v)
            for k, v in sanitized.entity_definitions.items()
        }

        # Filter examples with leaks
        sanitized.examples = [
            ex for ex in sanitized.examples
            if not self._check_text(ex.input_text) and not self._check_text(ex.expected_output)
        ]

        # Sanitize constraints
        sanitized.constraints = [
            self._sanitize_text(c) for c in sanitized.constraints
            if not self._check_text(c)
        ]

        sanitized.mutation_history.append("sanitized")

        return sanitized

    def _sanitize_text(self, text: str) -> str:
        """Remove leaked patterns from text."""
        result = text

        # Remove template markers
        for pattern in self.synpii_patterns:
            try:
                result = re.sub(pattern, "", result)
            except re.error:
                pass

        # Remove template strings
        for template in self.synpii_templates:
            result = result.replace(template, "")

        return result.strip()
