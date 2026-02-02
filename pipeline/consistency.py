"""Consistency Validator for NER Outputs.

Validates consistency of detected entities:
- Same text should have same type throughout document
- Entity boundaries should align with word boundaries
- Entity relationships should be consistent (e.g., PERSON in ORGANIZATION context)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re


@dataclass
class ConsistencyIssue:
    """A detected consistency issue."""
    issue_type: str  # "type_inconsistency", "boundary_issue", "relationship_issue"
    entity_text: str
    positions: List[Tuple[int, int]]  # (start, end) positions
    types_found: List[str]
    suggested_type: Optional[str] = None
    confidence: float = 0.0
    description: str = ""


@dataclass
class ValidationResult:
    """Result of consistency validation."""
    is_consistent: bool
    issues: List[ConsistencyIssue]
    corrected_entities: List[Dict]
    statistics: Dict = field(default_factory=dict)


class ConsistencyValidator:
    """Validates and corrects consistency issues in NER outputs.

    Checks for:
    1. Type consistency: Same text should have same type
    2. Boundary consistency: Entities should align with word/token boundaries
    3. Relationship consistency: Related entities should be consistent

    Example:
        validator = ConsistencyValidator()

        entities = [
            {"text": "Berlin", "entity_type": "LOCATION", "start": 10, "end": 16},
            {"text": "Berlin", "entity_type": "ORGANIZATION", "start": 50, "end": 56},
        ]

        result = validator.validate(entities, text)
        # Will flag "Berlin" as having inconsistent types
    """

    def __init__(
        self,
        enforce_word_boundaries: bool = True,
        min_type_consensus: float = 0.7,  # 70% agreement needed
        auto_correct: bool = True,
    ):
        """Initialize the consistency validator.

        Args:
            enforce_word_boundaries: Check that entities align with words
            min_type_consensus: Minimum ratio for type consensus
            auto_correct: Whether to automatically correct issues
        """
        self.enforce_word_boundaries = enforce_word_boundaries
        self.min_type_consensus = min_type_consensus
        self.auto_correct = auto_correct

        # Word boundary pattern (German-aware)
        self._word_boundary = re.compile(r'[\s\.,;:!\?\(\)\[\]\{\}"\'„"‚\'\-\–\—\/\\\\]')

    def validate(
        self,
        entities: List[Dict],
        text: str,
    ) -> ValidationResult:
        """Validate entities for consistency.

        Args:
            entities: List of entity dicts
            text: Full document text

        Returns:
            ValidationResult with issues and optionally corrected entities
        """
        issues = []

        # Check type consistency
        type_issues = self._check_type_consistency(entities)
        issues.extend(type_issues)

        # Check boundary consistency
        if self.enforce_word_boundaries:
            boundary_issues = self._check_boundary_consistency(entities, text)
            issues.extend(boundary_issues)

        # Check relationship consistency
        relationship_issues = self._check_relationship_consistency(entities, text)
        issues.extend(relationship_issues)

        # Auto-correct if enabled
        corrected = entities
        if self.auto_correct and issues:
            corrected = self._auto_correct(entities, issues)

        return ValidationResult(
            is_consistent=len(issues) == 0,
            issues=issues,
            corrected_entities=corrected,
            statistics={
                "total_entities": len(entities),
                "issues_found": len(issues),
                "type_inconsistencies": sum(1 for i in issues if i.issue_type == "type_inconsistency"),
                "boundary_issues": sum(1 for i in issues if i.issue_type == "boundary_issue"),
                "relationship_issues": sum(1 for i in issues if i.issue_type == "relationship_issue"),
            },
        )

    def _check_type_consistency(
        self,
        entities: List[Dict],
    ) -> List[ConsistencyIssue]:
        """Check that same text has consistent type."""
        issues = []

        # Group by normalized text
        text_to_entities: Dict[str, List[Dict]] = defaultdict(list)
        for entity in entities:
            text = entity.get("text", "").strip().lower()
            text_to_entities[text].append(entity)

        # Check each group
        for text, group in text_to_entities.items():
            if len(group) < 2:
                continue

            # Count types
            type_counts: Dict[str, int] = defaultdict(int)
            for entity in group:
                entity_type = entity.get("entity_type", entity.get("type", ""))
                type_counts[entity_type] += 1

            # Check for inconsistency
            if len(type_counts) > 1:
                total = sum(type_counts.values())
                majority_type = max(type_counts, key=type_counts.get)
                majority_ratio = type_counts[majority_type] / total

                # Only flag if no clear majority
                if majority_ratio < self.min_type_consensus:
                    issues.append(ConsistencyIssue(
                        issue_type="type_inconsistency",
                        entity_text=text,
                        positions=[(e["start"], e["end"]) for e in group],
                        types_found=list(type_counts.keys()),
                        suggested_type=majority_type if majority_ratio >= 0.5 else None,
                        confidence=majority_ratio,
                        description=f"'{text}' appears as {dict(type_counts)}",
                    ))

        return issues

    def _check_boundary_consistency(
        self,
        entities: List[Dict],
        text: str,
    ) -> List[ConsistencyIssue]:
        """Check that entities align with word boundaries."""
        issues = []

        for entity in entities:
            start = entity["start"]
            end = entity["end"]

            # Check start boundary
            start_ok = (
                start == 0 or
                self._word_boundary.match(text[start - 1]) is not None
            )

            # Check end boundary
            end_ok = (
                end >= len(text) or
                self._word_boundary.match(text[end]) is not None
            )

            if not start_ok or not end_ok:
                entity_text = text[start:end]
                issues.append(ConsistencyIssue(
                    issue_type="boundary_issue",
                    entity_text=entity_text,
                    positions=[(start, end)],
                    types_found=[entity.get("entity_type", entity.get("type", ""))],
                    description=f"Entity does not align with word boundaries: "
                               f"start_ok={start_ok}, end_ok={end_ok}",
                ))

        return issues

    def _check_relationship_consistency(
        self,
        entities: List[Dict],
        text: str,
    ) -> List[ConsistencyIssue]:
        """Check for relationship consistency issues."""
        issues = []

        # Check for PERSON appearing as part of email
        email_pattern = re.compile(r'[\w.+-]+@[\w.-]+\.\w+')
        for entity in entities:
            entity_type = entity.get("entity_type", entity.get("type", ""))
            entity_text = entity.get("text", text[entity["start"]:entity["end"]])

            if entity_type == "PERSON":
                # Check if this "person" is part of an email
                context_start = max(0, entity["start"] - 20)
                context_end = min(len(text), entity["end"] + 30)
                context = text[context_start:context_end]

                if email_pattern.search(context):
                    # Check if entity text is in the email
                    for match in email_pattern.finditer(context):
                        email_text = match.group()
                        if entity_text.lower() in email_text.lower():
                            issues.append(ConsistencyIssue(
                                issue_type="relationship_issue",
                                entity_text=entity_text,
                                positions=[(entity["start"], entity["end"])],
                                types_found=["PERSON"],
                                suggested_type="EMAIL_ADDRESS",
                                description=f"PERSON '{entity_text}' appears to be part of email",
                            ))

        # Check for LOCATION that is actually an ORGANIZATION suffix
        org_suffixes = ["Klinik", "Krankenhaus", "Hospital", "Praxis", "Zentrum"]
        for entity in entities:
            entity_type = entity.get("entity_type", entity.get("type", ""))
            entity_text = entity.get("text", text[entity["start"]:entity["end"]])

            if entity_type == "LOCATION":
                for suffix in org_suffixes:
                    if suffix in entity_text:
                        issues.append(ConsistencyIssue(
                            issue_type="relationship_issue",
                            entity_text=entity_text,
                            positions=[(entity["start"], entity["end"])],
                            types_found=["LOCATION"],
                            suggested_type="ORGANIZATION",
                            description=f"LOCATION '{entity_text}' contains organization indicator '{suffix}'",
                        ))
                        break

        return issues

    def _auto_correct(
        self,
        entities: List[Dict],
        issues: List[ConsistencyIssue],
    ) -> List[Dict]:
        """Auto-correct entities based on detected issues.

        Args:
            entities: Original entities
            issues: Detected issues

        Returns:
            Corrected entity list
        """
        corrected = []

        # Build correction map
        corrections: Dict[str, str] = {}  # text -> corrected type
        for issue in issues:
            if issue.suggested_type and issue.confidence >= 0.5:
                corrections[issue.entity_text.lower()] = issue.suggested_type

        # Apply corrections
        for entity in entities:
            entity_text = entity.get("text", "").strip().lower()
            corrected_entity = dict(entity)

            if entity_text in corrections:
                corrected_entity["entity_type"] = corrections[entity_text]
                corrected_entity["corrected"] = True
                corrected_entity["original_type"] = entity.get("entity_type", entity.get("type"))

            corrected.append(corrected_entity)

        return corrected

    def validate_against_expected(
        self,
        detected: List[Dict],
        expected: List[Dict],
    ) -> Dict:
        """Validate detected entities against expected ground truth.

        Args:
            detected: Detected entities
            expected: Expected entities (ground truth)

        Returns:
            Validation report
        """
        # Convert to comparable format
        detected_set = set()
        for e in detected:
            detected_set.add((
                e["start"],
                e["end"],
                e.get("entity_type", e.get("type", "")),
            ))

        expected_set = set()
        for e in expected:
            expected_set.add((
                e["start"],
                e["end"],
                e.get("entity_type", e.get("type", "")),
            ))

        # Calculate metrics
        true_positives = detected_set & expected_set
        false_positives = detected_set - expected_set
        false_negatives = expected_set - detected_set

        # Analyze type mismatches
        type_mismatches = []
        for det in detected_set - expected_set:
            for exp in expected_set - detected_set:
                if det[0] == exp[0] and det[1] == exp[1]:
                    # Same span, different type
                    type_mismatches.append({
                        "span": (det[0], det[1]),
                        "detected_type": det[2],
                        "expected_type": exp[2],
                    })

        return {
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "type_mismatches": type_mismatches,
            "precision": len(true_positives) / len(detected_set) if detected_set else 0.0,
            "recall": len(true_positives) / len(expected_set) if expected_set else 0.0,
        }
