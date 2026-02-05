"""Domain Rules Engine for German Clinical NER.

Applies domain-specific rules to:
- Suppress false positives (structural labels, medical terms)
- Reclassify entities based on context
- Validate entities against expected patterns
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union


class RuleAction(str, Enum):
    """Actions that rules can take on entities."""
    SUPPRESS = "suppress"           # Remove the entity
    RECLASSIFY = "reclassify"       # Change entity type
    BOOST = "boost"                 # Increase confidence
    PENALIZE = "penalize"           # Decrease confidence
    VALIDATE = "validate"           # Mark as validated
    FLAG = "flag"                   # Flag for review


@dataclass
class Rule:
    """A domain rule for entity processing.

    Rules can match entities by:
    - Pattern matching on entity text
    - Context matching (surrounding text)
    - Entity type matching
    - Custom predicates

    When matched, rules apply an action (suppress, reclassify, etc.).
    """
    name: str
    pattern: Optional[str] = None               # Regex pattern for entity text
    context_pattern: Optional[str] = None       # Regex pattern for surrounding context
    entity_types: Optional[List[str]] = None    # Entity types this rule applies to
    action: RuleAction = RuleAction.SUPPRESS
    target_type: Optional[str] = None           # For RECLASSIFY action
    confidence_delta: float = 0.0               # For BOOST/PENALIZE actions
    priority: int = 0                           # Higher priority rules apply first
    enabled: bool = True
    description: str = ""

    _compiled_pattern: Optional[Pattern] = field(default=None, repr=False)
    _compiled_context: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        """Compile regex patterns."""
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        if self.context_pattern:
            self._compiled_context = re.compile(self.context_pattern, re.IGNORECASE)

    def matches(
        self,
        entity_text: str,
        entity_type: str,
        context: str = "",
    ) -> bool:
        """Check if this rule matches an entity.

        Args:
            entity_text: The entity text
            entity_type: The entity type
            context: Surrounding context text

        Returns:
            True if rule matches
        """
        if not self.enabled:
            return False

        # Check entity type
        if self.entity_types and entity_type not in self.entity_types:
            return False

        # Check pattern
        if self._compiled_pattern:
            if not self._compiled_pattern.search(entity_text):
                return False

        # Check context
        if self._compiled_context:
            if not self._compiled_context.search(context):
                return False

        return True


# Default clinical rules for German medical text
DEFAULT_CLINICAL_RULES = [
    # Suppress structural labels that look like entities
    Rule(
        name="suppress_structural_labels",
        pattern=r"^(Behandelnder Arzt|Aufnahmedatum|Geburtsdatum|Entlassdatum|Stationärer Aufenthalt|Fallnummer|Patientennummer|Diagnose|Anamnese|Befund|Therapie|Epikrise|Medikation)$",
        action=RuleAction.SUPPRESS,
        priority=100,
        description="Suppress structural document labels",
    ),

    # Suppress medical terms that look like locations
    Rule(
        name="suppress_medical_location_terms",
        pattern=r"^(TIMI-\d+-Fluss|Arteria|Vena|Bulbus|Kortex|Medulla|Lumen|Foramen|Atrium|Ventrikel|Apex|Basis|Corpus|Caput)$",
        entity_types=["LOCATION"],
        action=RuleAction.SUPPRESS,
        priority=90,
        description="Suppress medical anatomy terms misclassified as locations",
    ),

    # Suppress "Patient" as occupation
    Rule(
        name="suppress_patient_occupation",
        pattern=r"^Patient(in)?$",
        entity_types=["OCCUPATION"],
        action=RuleAction.SUPPRESS,
        priority=90,
        description="'Patient' is not an occupation",
    ),

    # Suppress common false positive ages
    Rule(
        name="suppress_false_ages",
        pattern=r"^(Station|Zimmer|Bett|Stufe|Grad|Phase)\s*\d+",
        entity_types=["AGE"],
        action=RuleAction.SUPPRESS,
        priority=85,
        description="Suppress room/bed numbers misclassified as age",
    ),

    # Reclassify email substrings
    Rule(
        name="reclassify_email_person",
        pattern=r"[\w.]+@[\w.]+",
        entity_types=["PERSON"],
        action=RuleAction.RECLASSIFY,
        target_type="EMAIL_ADDRESS",
        priority=80,
        description="Email addresses should not be PERSON",
    ),

    # Boost confidence for entities with context clues
    Rule(
        name="boost_person_with_title",
        context_pattern=r"(Herr|Frau|Dr\.|Prof\.)\s*$",
        entity_types=["PERSON"],
        action=RuleAction.BOOST,
        confidence_delta=0.1,
        priority=50,
        description="Boost PERSON confidence when preceded by title",
    ),

    # Boost occupation in social history context
    Rule(
        name="boost_occupation_social",
        context_pattern=r"(arbeitet|beruflich|tätig|Beruf|beschäftigt)",
        entity_types=["OCCUPATION"],
        action=RuleAction.BOOST,
        confidence_delta=0.15,
        priority=50,
        description="Boost OCCUPATION in social history context",
    ),

    # Penalize location that looks like organization
    Rule(
        name="penalize_clinic_location",
        pattern=r"(Klinik|Krankenhaus|Praxis|Zentrum|Institut)",
        entity_types=["LOCATION"],
        action=RuleAction.PENALIZE,
        confidence_delta=-0.2,
        priority=40,
        description="Clinics/hospitals should be ORGANIZATION, not LOCATION",
    ),

    # Suppress dates that are just years
    Rule(
        name="suppress_year_only_date",
        pattern=r"^\d{4}$",
        entity_types=["DATE_TIME"],
        action=RuleAction.SUPPRESS,
        priority=30,
        description="Single year numbers are often not meaningful dates",
    ),

    # Flag unusual postal codes for review
    Rule(
        name="flag_unusual_postal",
        pattern=r"^[0-3]\d{4}$",  # East German codes start with 0-3
        entity_types=["DE_POSTAL_CODE"],
        action=RuleAction.FLAG,
        priority=20,
        description="Flag East German postal codes for review",
    ),
]


@dataclass
class RuleApplication:
    """Record of a rule being applied to an entity."""
    rule_name: str
    action: RuleAction
    entity_text: str
    entity_type: str
    original_confidence: float
    new_confidence: Optional[float] = None
    new_type: Optional[str] = None


class DomainRulesEngine:
    """Engine for applying domain-specific rules to detected entities.

    The rules engine processes entities through a prioritized set of rules,
    applying actions like suppression, reclassification, and confidence
    adjustment based on domain knowledge.

    Example:
        engine = DomainRulesEngine()

        # Process an entity
        result = engine.apply_rules(
            entity_text="Behandelnder Arzt",
            entity_type="PERSON",
            confidence=0.8,
            context="Behandelnder Arzt: Dr. Schmidt"
        )

        if result is None:
            print("Entity was suppressed")
    """

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        use_defaults: bool = True,
    ):
        """Initialize the rules engine.

        Args:
            rules: Custom rules to use
            use_defaults: Whether to include default clinical rules
        """
        self.rules: List[Rule] = []

        if use_defaults:
            self.rules.extend(DEFAULT_CLINICAL_RULES)

        if rules:
            self.rules.extend(rules)

        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: -r.priority)

        self.application_log: List[RuleApplication] = []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule and re-sort by priority."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Returns:
            True if rule was found and removed
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False

    def apply_rules(
        self,
        entity_text: str,
        entity_type: str,
        confidence: float,
        context: str = "",
        log_applications: bool = True,
    ) -> Optional[Dict]:
        """Apply rules to an entity.

        Args:
            entity_text: The entity text
            entity_type: The entity type
            confidence: Detection confidence
            context: Surrounding context text
            log_applications: Whether to log rule applications

        Returns:
            Dict with processed entity, or None if suppressed
            Contains: entity_type, confidence, flagged, reclassified
        """
        current_type = entity_type
        current_confidence = confidence
        flagged = False
        reclassified = False

        for rule in self.rules:
            if rule.matches(entity_text, current_type, context):
                if rule.action == RuleAction.SUPPRESS:
                    if log_applications:
                        self.application_log.append(RuleApplication(
                            rule_name=rule.name,
                            action=rule.action,
                            entity_text=entity_text,
                            entity_type=entity_type,
                            original_confidence=confidence,
                        ))
                    return None

                elif rule.action == RuleAction.RECLASSIFY:
                    if rule.target_type:
                        if log_applications:
                            self.application_log.append(RuleApplication(
                                rule_name=rule.name,
                                action=rule.action,
                                entity_text=entity_text,
                                entity_type=entity_type,
                                original_confidence=confidence,
                                new_type=rule.target_type,
                            ))
                        current_type = rule.target_type
                        reclassified = True

                elif rule.action == RuleAction.BOOST:
                    new_conf = min(1.0, current_confidence + rule.confidence_delta)
                    if log_applications:
                        self.application_log.append(RuleApplication(
                            rule_name=rule.name,
                            action=rule.action,
                            entity_text=entity_text,
                            entity_type=entity_type,
                            original_confidence=confidence,
                            new_confidence=new_conf,
                        ))
                    current_confidence = new_conf

                elif rule.action == RuleAction.PENALIZE:
                    new_conf = max(0.0, current_confidence + rule.confidence_delta)
                    if log_applications:
                        self.application_log.append(RuleApplication(
                            rule_name=rule.name,
                            action=rule.action,
                            entity_text=entity_text,
                            entity_type=entity_type,
                            original_confidence=confidence,
                            new_confidence=new_conf,
                        ))
                    current_confidence = new_conf

                elif rule.action == RuleAction.FLAG:
                    flagged = True
                    if log_applications:
                        self.application_log.append(RuleApplication(
                            rule_name=rule.name,
                            action=rule.action,
                            entity_text=entity_text,
                            entity_type=entity_type,
                            original_confidence=confidence,
                        ))

        return {
            "entity_type": current_type,
            "confidence": current_confidence,
            "flagged": flagged,
            "reclassified": reclassified,
        }

    def process_entities(
        self,
        entities: List[Dict],
        text: str,
        context_window: int = 50,
    ) -> List[Dict]:
        """Process a list of entities through the rules engine.

        Args:
            entities: List of entity dicts with text, entity_type, start, end, score
            text: Full document text (for context extraction)
            context_window: Characters of context on each side

        Returns:
            Filtered and processed entity list
        """
        processed = []

        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", len(text))

            # Extract context
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]

            result = self.apply_rules(
                entity_text=entity.get("text", text[start:end]),
                entity_type=entity.get("entity_type", entity.get("type")),
                confidence=entity.get("score", 0.5),
                context=context,
            )

            if result is not None:
                processed_entity = dict(entity)
                processed_entity["entity_type"] = result["entity_type"]
                processed_entity["score"] = result["confidence"]
                if result["flagged"]:
                    processed_entity["flagged"] = True
                if result["reclassified"]:
                    processed_entity["reclassified"] = True
                processed.append(processed_entity)

        return processed

    def get_statistics(self) -> Dict:
        """Get statistics about rule applications."""
        by_rule = {}
        by_action = {}

        for app in self.application_log:
            by_rule[app.rule_name] = by_rule.get(app.rule_name, 0) + 1
            by_action[app.action.value] = by_action.get(app.action.value, 0) + 1

        return {
            "total_applications": len(self.application_log),
            "by_rule": by_rule,
            "by_action": by_action,
        }

    def clear_log(self) -> None:
        """Clear the application log."""
        self.application_log = []
