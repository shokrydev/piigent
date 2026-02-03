# PIIgent: Agentic Weakness Discovery for PII Detection

A **LangGraph-based system** for systematically discovering and explaining anonymization failures in German clinical text. PIIgent uses iterative hypothesis-test loops and self-critique mechanisms—drawing on ideas from recent agentic AI research—to understand *why* PII detection pipelines fail.

**This is a research exploration tool**, not a production anonymizer.

## The Problem: Structural Failure in PII Detection

Empirical testing shows that even well-tuned PII detection pipelines achieve only **76-78% recall** on German clinical text. That means **22-24% of PII survives anonymization**.

This gap isn't due to bad components:
- Pattern recognizers achieve ~99% F2 on structured German IDs (KVNR, LANR, PLZ)
- LLMs achieve ~79% F2 on contextual entities (names, locations)
- **The problem is structural**: naive single-pass pipelines with max-score aggregation fail in predictable ways

Traditional NER pipelines make single-pass decisions with fixed rules. But clinical text has **overlapping entities** (PLZ vs LOCATION compete for the same span), **context dependencies** (names require honorific prefixes), and **format variations** (seperators and spaces in IDs). A single pass cannot sufficiently reason about these conflicts.

### Why an Iterative, Agentic Approach?

An iterative system with specialized components can do what static pipelines cannot:
- **Reason** about why entity conflicts occur (hypothesis generation)
- **Act** by generating targeted test cases to validate hypotheses
- **Observe** results and **reflect** on root causes
- **Iterate** to refine understanding

This draws on patterns from agentic AI research:

| Concept | How PIIgent Uses It | Inspiration |
|---------|---------------------|-------------|
| **Reasoning + Acting** | Weakness Analyzer interleaves hypothesis generation with targeted test case creation | ReAct [Yao et al., 2022] |
| **Self-Critique** | Agents generate explanations for failures and propose fixes | Reflexion [Shinn et al., 2023] |
| **Ensemble Coordination** | Specialized recognizers (pattern-based, LLM-based) with orchestrated aggregation | Multi-agent systems [Han et al., 2024] |
| **Curriculum Learning** | Difficulty-aware progression with failure-weighted sampling | Bengio et al., 2009 |

## Architecture

### Recognizer Ensemble

PIIgent combines two types of recognizers, each with different strengths:

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

All recognizers come from the [custom Presidio fork](https://github.com/shokrydev/anoner).

### Two Workflows

**1. Privacy Pipeline** — Document processing with ensemble aggregation

```
                ┌─────────────────────────┐
                │  Detection Coordinator  │
                │  ┌────────┐ ┌─────────┐ │
                │  │Pattern │ │  LLM    │ │  ← Parallel recognizers
                │  │  (8x)  │ │(Ollama) │ │
                │  └───┬────┘ └───┬─────┘ │
                │      └────┬─────┘       │
                │           ▼             │
                │   Confidence Scoring    │  ← Aggregation with conflict resolution
                └──────────┬──────────────┘
                           │
                    ┌──────┴──────┐
            conf<θ? │             │ conf≥θ?
                    ▼             ▼
            ┌─────────────┐ ┌─────────────────┐
            │ HITL Review │ │  Anonymization  │
            └──────┬──────┘ └────────┬────────┘
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │ Quality Auditor │  ← Re-detection for leakage check
                   └─────────────────┘
```

**2. Weakness Analyzer** — Iterative exploration loop

The Weakness Analyzer follows a hypothesis-test cycle inspired by ReAct's interleaving of reasoning and acting:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEAKNESS ANALYZER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐                                                     │
│  │ Benchmark      │   Generate synthetic docs, run pipeline,            │
│  │ Pipeline       │   calculate per-entity-type metrics                 │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐   Identify entity types with low recall:            │
│  │ Identify       │   • OVERLAP_CONFLICT (PLZ vs LOCATION)              │
│  │ Weaknesses     │   • COVERAGE_GAP (AGE not detected)                 │
│  └───────┬────────┘   • CONTEXT_DEPENDENCY (needs "Herr/Frau")          │
│          │                                                              │
│          ▼                                        ┌─────────────────┐   │
│  ┌────────────────┐   Generate hypothesis:        │                 │   │
│  │ Hypothesize    │   "PLZ detected but LOCATION  │   REASON        │   │
│  │                │    overlaps with higher score"│                 │   │
│  └───────┬────────┘                               └─────────────────┘   │
│          │                                                              │
│          ▼                                        ┌─────────────────┐   │
│  ┌────────────────┐   Create targeted test cases: │                 │   │
│  │ Test           │   • "10115" (PLZ alone)       │   ACT           │   │
│  │                │   • "10115 Berlin" (combined) │                 │   │
│  └───────┬────────┘   • "PLZ: 10115" (contextual) └─────────────────┘   │
│          │                                                              │
│          ▼                                        ┌─────────────────┐   │
│  ┌────────────────┐   Analyze results:            │                 │   │
│  │ Root Cause     │   "LLM's LOCATION score       │   OBSERVE       │   │
│  │ Analysis       │    beats pattern's PLZ score" │                 │   │
│  └───────┬────────┘                               └─────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐                                                     │
│  │ Self-Critique  │◀──── Loop: explain failure, propose fix,            │
│  │ Agents         │       continue to next weakness                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐                                                     │
│  │ Report         │   Evidence-backed recommendations                   │
│  └────────────────┘                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Illustrative Output** (showing the type of analysis produced):
```
Weakness: DE_POSTAL_CODE — low recall despite pattern match
Hypothesis: Overlap with LOCATION entity type
Test cases generated: 15 variations of PLZ with/without city names
Finding: LLM's LOCATION label consistently overrides pattern's
         DE_POSTAL_CODE due to higher confidence scores
Root cause: Max-score aggregation ignores entity specificity
Recommendation: Implement PREFER_SPECIFIC resolution strategy
```

### Weakness Taxonomy

The analyzer categorizes failures into actionable types:

| Type | Description | Example |
|------|-------------|---------|
| `OVERLAP_CONFLICT` | Higher-score entity replaces correct detection | LOCATION "53168 Bonn" beats DE_POSTAL_CODE "53168" |
| `COVERAGE_GAP` | No recognizer detects this entity type | AGE patterns like "64-jährige" unrecognized |
| `CONTEXT_DEPENDENCY` | Detection requires specific context words | "Müller" only detected with "Herr" prefix |
| `FORMAT_VARIATION` | Entity format differs from expected pattern | KVNR with spaces "A 123 456 780" |
| `ENTITY_CONFUSION` | Wrong entity type assigned | BSNR detected as LANR (same format) |
| `LEAKAGE` | PII survives anonymization | Partial name remains after full name anonymized |

## Key Findings

From running PIIgent's weakness exploration on synthetic German clinical documents, the following patterns emerged:

1. **Overlap conflicts are the primary failure mode** — When PLZ appears adjacent to a city name ("10115 Berlin"), the LLM labels the combined span as LOCATION, overriding the pattern recognizer's more specific DE_POSTAL_CODE detection. Max-score aggregation favors the LLM's higher confidence.

2. **Pattern recognizers are highly reliable when not overridden** — KVNR, LANR, BSNR achieve ~99% F2 thanks to checksum validation. The failures occur at the aggregation stage, not detection.

3. **Context dependency causes false negatives** — Standalone surnames ("Müller") are often missed; with honorifics ("Herr Müller") they're detected. The LLM needs contextual cues for German names.

4. **Coverage gaps exist** — AGE patterns ("64-jährige") have no dedicated recognizer. This is a gap in the recognizer ensemble, not an aggregation conflict.

These findings suggest targeted fixes: implement PREFER_SPECIFIC aggregation for structured IDs, add context boosting for names, create AGE pattern recognizer.

## Key Components

### 1. Self-Critique Agents

The system includes agents that generate natural language explanations for failures:

- **RationaleAgent** — Explains why an entity was tagged or missed
- **ErrorTaxonomyAgent** — Classifies errors into systematic categories
- **FixProposalAgent** — Proposes specific changes based on error patterns
- **VerificationAgent** — Tests whether proposed fixes improve performance

These agents generate explanations and proposals; applying fixes requires human review or additional automation.

### 2. Prompt Evolution System

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

### 3. Curriculum Learning with Failure Weighting

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

**Failure-weighted sampling** oversamples regions where the system fails—starting with simple cases before progressing to complex ones.

### 4. Multi-Dimensional Evaluation

Beyond aggregate F1, PIIgent tracks metrics that reveal specific failure modes:

| Category | Metrics |
|----------|---------|
| **Span Matching** | Exact P/R/F1, Partial P/R/F1 (50% overlap threshold) |
| **Type Matching** | Type-only P/R/F1 (ignore boundary errors) |
| **Boundary Analysis** | Off-by-one errors, partial span rate |
| **Calibration** | Expected Calibration Error (ECE), overconfidence rate |
| **Per-Entity-Type** | Confusion matrix, entity-specific F2 |
| **Multi-Recognizer** | Agreement rate, disagreement analysis |

**Regression testing** with golden test sets prevents improvements in one area from causing regressions elsewhere.

## Quick Start

### Installation

```bash
git clone https://github.com/shokrydev/piigent.git
cd piigent

python -m venv .venv
source .venv/bin/activate

# Install custom Presidio fork (required)
pip install -e ./anoner/presidio-analyzer

# Install pipeline dependencies
pip install langgraph langchain-core httpx

# Optional: Pull model for LLM recognition
ollama pull ministral
```

### Run Pipeline

```python
from graph.pipeline_graph import run_pipeline

result = run_pipeline(
    document="""
    Entlassungsbrief - Charité Berlin
    Patient: Max Mustermann
    KVNR: A123456780
    PLZ: 10117 Berlin
    """,
    confidence_threshold=0.7,
    human_in_loop=False,
    preset='clinical',
)

print(result['anonymized_text'])
```

### Run Weakness Analysis

```python
from agents.weakness_analyzer import analyze_weaknesses

report = analyze_weaknesses(
    num_docs=20,
    max_rounds=3,
)

print(report["recommendations"])
```

## Current Limitations

| Issue | Impact | Status |
|-------|--------|--------|
| ~76-78% recall | 22-24% of PII survives | Core research problem |
| Overlap conflicts | Pattern detections overridden by LLM | Identified, fix proposed |
| No AGE detection | Age patterns unrecognized | Coverage gap |
| Context sensitivity | Names need honorific context | Identified |
| Synthetic data only | No real-world validation yet | Limitation |

**Do not use this for production GDPR/DSGVO compliance.** The purpose is to discover and document failure modes, not to provide robust anonymization.

## Project Structure

```
piigent/
├── agents/                         # Detection and self-critique agents
│   ├── detection_coordinator.py    # Multi-recognizer orchestration
│   ├── aggregator.py               # Entity deduplication/merging
│   ├── weakness_analyzer.py        # Iterative weakness exploration
│   ├── rationale.py                # Explains predictions
│   ├── error_taxonomy.py           # Classifies errors
│   ├── fix_proposal.py             # Proposes fixes
│   └── verification.py             # Verifies fix effectiveness
├── prompts/                        # Prompt Evolution System
│   ├── genome.py                   # PromptGenotype dataclass
│   ├── store.py                    # SQLite-backed lineage storage
│   ├── mutations.py                # Mutation operators
│   ├── crossover.py                # Crossover operations
│   ├── fitness.py                  # Fitness tracking
│   └── llm_mutation.py             # LLM-guided mutation
├── evaluation/                     # Multi-Dimensional Evaluation
│   ├── metrics.py                  # Beyond P/R/F1
│   ├── regression.py               # Regression prevention
│   ├── disagreement.py             # Multi-recognizer disagreement
│   └── distribution.py             # Distribution matching
├── curriculum/                     # Curriculum Learning
│   ├── difficulty.py               # Difficulty dimensions
│   ├── controller.py               # Auto-promotion/demotion
│   └── sampler.py                  # Failure-weighted sampling
├── pipeline/                       # Multi-Step NER Pipeline
│   ├── section_parser.py           # German clinical sections
│   ├── domain_rules.py             # Clinical domain rules
│   ├── overlap_resolver.py         # PREFER_SPECIFIC resolution
│   └── consistency.py              # Consistency validation
├── safeguards/                     # System Safeguards
│   ├── overfitting.py              # Overfitting detection
│   └── leakage.py                  # Template leakage checking
├── wrappers/                       # Integration Wrappers
│   ├── llm_wrapper.py              # PromptManagedLLM
│   └── synpii_wrapper.py           # Curriculum-aware SynPII
├── ui/                             # Human-in-the-Loop
│   └── hitl_interface.py           # Gradio HITL interface
├── graph/
│   ├── pipeline_graph.py           # LangGraph workflow
│   └── state.py                    # PipelineState, DetectedEntity
├── tests/
│   └── test_advanced_architecture.py
├── anoner/                         # Custom Presidio fork (submodule)
└── synpii/                         # Synthetic PII generator (submodule)
```

## Future Directions

1. **Real-world validation** — Test on actual clinical documents (with appropriate ethics approval)

2. **Automated fix application** — Close the loop from proposed fixes to implementation

3. **Cross-domain transfer** — Do discovered weaknesses generalize to non-clinical German text?

4. **Active learning** — Use disagreement analysis to prioritize human annotation

## References

### Agentic AI Research

- **ReAct**: Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

- **Reflexion**: Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

- **Multi-Agent Systems**: Han, J., et al. (2024). "LLM Multi-Agent Systems: Challenges and Open Problems." [arXiv:2402.03578](https://arxiv.org/abs/2402.03578)

- **Curriculum Learning**: Bengio, Y., et al. (2009). "Curriculum Learning." *ICML 2009*.

### Tools & Frameworks

- [anoner](https://github.com/shokrydev/anoner) — Custom Presidio fork with German healthcare recognizers
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Workflow orchestration framework

## Author

[@shokrydev](https://github.com/shokrydev)
