# Architecture Details

## Recognizer Ensemble

PIIgent combines two types of recognizers:

**Pattern-Based Recognizers** — High precision on structured German identifiers. Each recognizer targets one entity type with format validation:

| Recognizer | Entity Type | Validation |
|------------|-------------|------------|
| `DeKvnrRecognizer` | DE_KVNR (health insurance ID) | Luhn checksum |
| `DeLanrRecognizer` | DE_LANR (physician ID) | KBV checksum |
| `DeBsnrRecognizer` | DE_BSNR (facility ID) | KV code validation |
| `DeTelematikIdRecognizer` | DE_TELEMATIK_ID | Format validation |
| `DePersonalIdRecognizer` | DE_PERSONAL_ID | Weighted checksum |
| `DeTaxIdRecognizer` | DE_TAX_ID | ISO 7064 |
| `DeSocialSecurityRecognizer` | DE_SOCIAL_SECURITY | Checksum algorithm |
| `DePostalCodeRecognizer` | DE_POSTAL_CODE | Range validation |

**LLM-Based Recognizer** — Contextual detection of entities that lack fixed patterns:

| Recognizer | Entity Types | Notes |
|------------|--------------|-------|
| `OllamaNERecognizer` | PERSON, LOCATION, DATE_TIME, ORGANIZATION, etc. | Via Ollama (Ministral) |

## Self-Critique Agents

The system includes agents that generate natural language explanations for failures:

- **RationaleAgent** — Explains why an entity was tagged or missed
- **ErrorTaxonomyAgent** — Classifies errors into systematic categories
- **FixProposalAgent** — Proposes specific changes based on error patterns
- **VerificationAgent** — Tests whether proposed fixes improve performance

## Prompt Evolution System

A genetic approach to prompt optimization—treating prompts as genotypes that evolve through mutation, crossover, and fitness selection:

```python
@dataclass
class PromptGenotype:
    instruction_block: str
    entity_definitions: Dict[str, str]
    examples: List[PromptExample]
    constraints: List[str]

    # Lineage tracking
    parent_ids: List[str]
    generation: int
    mutation_history: List[str]
    fitness_scores: Dict[str, float]
```

**Mutation operators**: `ADD_EXAMPLE`, `SWAP_EXAMPLE`, `ADD_CONSTRAINT`, `EMPHASIZE_ENTITY`, `ADD_PATTERN`

## Curriculum Learning

Structured difficulty progression:

```python
@dataclass
class DifficultyDimensions:
    entity_count: int        # 1-5: Entities per document
    entity_variety: int      # 1-5: Different entity types
    format_variation: int    # 1-5: Non-standard formats
    overlap_frequency: int   # 1-5: Overlapping entities
    context_ambiguity: int   # 1-5: Ambiguous contexts
```

**Failure-weighted sampling** oversamples regions where the system fails.

## Evaluation Metrics

Beyond aggregate F1, PIIgent tracks:

| Category | Metrics |
|----------|---------|
| **Span Matching** | Exact P/R/F1, Partial P/R/F1 (50% overlap threshold) |
| **Type Matching** | Type-only P/R/F1 (ignore boundary errors) |
| **Boundary Analysis** | Off-by-one errors, partial span rate |
| **Calibration** | Expected Calibration Error (ECE), overconfidence rate |
| **Per-Entity-Type** | Confusion matrix, entity-specific F2 |
| **Multi-Recognizer** | Agreement rate, disagreement analysis |
