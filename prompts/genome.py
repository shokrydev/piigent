"""Prompt Genotype - Structured prompt representation for genetic operations.

The PromptGenotype class represents prompts as structured, evolvable units
that can be systematically mutated, crossed over, and tracked through lineage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import json


@dataclass
class PromptExample:
    """A few-shot example in the prompt.

    Each example demonstrates how to identify a specific entity type,
    providing the model with concrete input/output pairs.
    """
    input_text: str
    expected_output: str
    entity_type: str  # Which entity this example demonstrates
    source: str = "manual"  # "manual", "high_confidence", "human_labeled"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "entity_type": self.entity_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptExample":
        """Create from dictionary."""
        return cls(
            input_text=data["input_text"],
            expected_output=data["expected_output"],
            entity_type=data["entity_type"],
            source=data.get("source", "manual"),
        )


@dataclass
class PromptGenotype:
    """Structured prompt representation for genetic operations.

    A prompt genome consists of several components that can be
    independently mutated and combined:

    - instruction_block: Core instructions for the model
    - entity_definitions: Per-entity-type definitions and patterns
    - examples: Few-shot examples demonstrating each entity type
    - constraints: Explicit constraints ("Never tag structural labels")
    - output_schema: JSON schema defining expected output format

    The genome also tracks its lineage (parent_ids, generation) and
    fitness scores from evaluations.
    """
    id: str

    # Genetic components
    instruction_block: str
    entity_definitions: Dict[str, str] = field(default_factory=dict)
    examples: List[PromptExample] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_schema: str = ""

    # Lineage tracking
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    mutation_history: List[str] = field(default_factory=list)

    # Fitness tracking
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    eval_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id(prefix: str = "genome") -> str:
        """Generate a unique genome ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:6]
        return f"{prefix}_{timestamp}_{random_suffix}"

    @classmethod
    def create_default(cls) -> "PromptGenotype":
        """Create a default genome with baseline prompts.

        This provides a good starting point for evolution, based on
        the prompts that work reasonably well for German clinical text.
        """
        return cls(
            id=cls.generate_id(),
            instruction_block="""You are a PII extraction system. Extract ALL personally identifiable information from the text.

CRITICAL: Find EVERY occurrence of each entity. If a name appears 5 times, return 5 separate entries.

Return a JSON array with ALL entities found. Each entity needs:
- text: exact text span as it appears in the original text
- label: entity type (use the exact labels provided)
- start: character start position (0-indexed)
- end: character end position

Return ONLY the JSON array. Be exhaustive - missing PII is worse than false positives.""",
            entity_definitions={
                "PERSON": "Full names, titles with surnames (Herr/Frau/Dr.), individual people",
                "LOCATION": "Streets, cities, hospital wards, geographic places",
                "ORGANIZATION": "Companies, hospitals, clinics, insurance providers",
                "AGE": "Age expressions: '63-jährig', '44-jährige', '51 Jahre alt'",
                "DATE_TIME": "Dates and times: '23.09.1990', 'am 17.04.2019'",
                "OCCUPATION": "Jobs/professions: 'Lehrer', 'Ingenieurin', 'Elektriker'",
                "DE_KVNR": "German health insurance number (10 digits)",
                "DE_POSTAL_CODE": "German postal codes (5 digits)",
            },
            examples=[
                PromptExample(
                    input_text="Herr Bauer, 63-jährig, wohnhaft in Berlin.",
                    expected_output='[{"text": "Herr Bauer", "label": "PERSON", "start": 0, "end": 10}, {"text": "63-jährig", "label": "AGE", "start": 12, "end": 21}, {"text": "Berlin", "label": "LOCATION", "start": 36, "end": 42}]',
                    entity_type="PERSON",
                    source="manual",
                ),
                PromptExample(
                    input_text="Die Patientin arbeitet als Lehrerin in München.",
                    expected_output='[{"text": "Lehrerin", "label": "OCCUPATION", "start": 27, "end": 35}, {"text": "München", "label": "LOCATION", "start": 39, "end": 46}]',
                    entity_type="OCCUPATION",
                    source="manual",
                ),
            ],
            constraints=[
                "IMPORTANT: Find ALL occurrences of each entity in the text.",
                "Do NOT tag structural labels like 'Behandelnder Arzt' or 'Aufnahmedatum' as entities.",
                "Medical terms like 'Arteria', 'TIMI-3-Fluss' are NOT locations.",
                "'Patient' or 'Patientin' alone is NOT an occupation.",
            ],
            output_schema='{"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "label": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["text", "label", "start", "end"]}}',
            generation=0,
        )

    def build_system_prompt(self) -> str:
        """Build the complete system prompt from genome components.

        Returns:
            The full system prompt assembled from instruction_block,
            entity_definitions, and constraints.
        """
        parts = [self.instruction_block]

        if self.entity_definitions:
            parts.append("\nEntity Types to Extract:")
            for entity_type, definition in self.entity_definitions.items():
                parts.append(f"- {entity_type}: {definition}")

        if self.constraints:
            parts.append("\nConstraints:")
            for constraint in self.constraints:
                parts.append(f"- {constraint}")

        return "\n".join(parts)

    def build_german_additions(self) -> str:
        """Build German-specific prompt additions.

        This generates the language_additions["de"] content for
        OllamaNERecognizer from the genome's examples and definitions.
        """
        parts = ["\n\nGerman-specific patterns to find:"]

        # Add entity-specific patterns from definitions
        for entity_type, definition in self.entity_definitions.items():
            parts.append(f"- {entity_type}: {definition}")

        # Add examples section
        if self.examples:
            parts.append("\nExamples:")
            for example in self.examples[:5]:  # Limit to 5 examples
                parts.append(f"Input: \"{example.input_text}\"")
                parts.append(f"Output: {example.expected_output}")

        return "\n".join(parts)

    def fitness_for_entity(self, entity_type: str) -> float:
        """Get fitness score for a specific entity type.

        Args:
            entity_type: The entity type to get fitness for

        Returns:
            Fitness score (0.0 if not evaluated for this type)
        """
        key = f"f1_{entity_type.lower()}"
        return self.fitness_scores.get(key, 0.0)

    def overall_fitness(self) -> float:
        """Get overall fitness (average F1 across entity types)."""
        f1_scores = [v for k, v in self.fitness_scores.items() if k.startswith("f1_")]
        if not f1_scores:
            return 0.0
        return sum(f1_scores) / len(f1_scores)

    def add_example(self, example: PromptExample) -> None:
        """Add a new example to the genome."""
        self.examples.append(example)
        self.mutation_history.append(f"add_example:{example.entity_type}")

    def add_constraint(self, constraint: str) -> None:
        """Add a new constraint to the genome."""
        self.constraints.append(constraint)
        self.mutation_history.append(f"add_constraint:{constraint[:30]}")

    def update_entity_definition(self, entity_type: str, definition: str) -> None:
        """Update or add an entity definition."""
        self.entity_definitions[entity_type] = definition
        self.mutation_history.append(f"update_definition:{entity_type}")

    def clone(self, new_id: Optional[str] = None) -> "PromptGenotype":
        """Create a copy of this genome with a new ID.

        Args:
            new_id: ID for the new genome (auto-generated if None)

        Returns:
            A new PromptGenotype with copied content
        """
        return PromptGenotype(
            id=new_id or self.generate_id(),
            instruction_block=self.instruction_block,
            entity_definitions=dict(self.entity_definitions),
            examples=list(self.examples),
            constraints=list(self.constraints),
            output_schema=self.output_schema,
            parent_ids=[self.id],
            generation=self.generation + 1,
            mutation_history=[],
            fitness_scores={},
            eval_count=0,
            created_at=datetime.now(),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "instruction_block": self.instruction_block,
            "entity_definitions": self.entity_definitions,
            "examples": [e.to_dict() for e in self.examples],
            "constraints": self.constraints,
            "output_schema": self.output_schema,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "mutation_history": self.mutation_history,
            "fitness_scores": self.fitness_scores,
            "eval_count": self.eval_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptGenotype":
        """Create from dictionary."""
        examples = [PromptExample.from_dict(e) for e in data.get("examples", [])]
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data["id"],
            instruction_block=data["instruction_block"],
            entity_definitions=data.get("entity_definitions", {}),
            examples=examples,
            constraints=data.get("constraints", []),
            output_schema=data.get("output_schema", ""),
            parent_ids=data.get("parent_ids", []),
            generation=data.get("generation", 0),
            mutation_history=data.get("mutation_history", []),
            fitness_scores=data.get("fitness_scores", {}),
            eval_count=data.get("eval_count", 0),
            created_at=created_at,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "PromptGenotype":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
