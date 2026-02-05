# PIIgent: Agentic Weakness Discovery for PII Detection

A **LangGraph-based system** for systematically discovering and explaining anonymization failures in German clinical text. PIIgent uses iterative hypothesis-test loops and self-critique mechanisms to understand *why* PII detection flows fail.

**This is a research exploration tool**, not a production anonymizer.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for installation and usage instructions.

You can also run the included flow demos:
```bash
# Simplified run commands (uv handles dependencies automatically)
uv run demo/run_agentic_flow.py
uv run demo/run_weakness_analysis.py
```

## The Problem

Empirical testing shows that even well-tuned PII detection flows achieve only ~76-78% recall on German clinical text. The problem is structural: naive single-pass flows cannot resolve conflicts between overlapping entities (e.g., ZIP codes vs. Locations) or handle subtle context dependencies.

PIIgent uses an iterative, agentic approach to:
1. **Reason** about entity conflicts
2. **Act** by generating targeted test cases
3. **Observe** results and identify root causes
4. **Evolve** prompts to fix the weaknesses

For detailed research findings, see [docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md).

## Architecture

PIIgent combines specialized pattern recognizers with LLM-based contextual understanding, orchestrated by a LangGraph workflow.

### 1. Privacy Flow
Standard document processing with ensemble aggregation and conflict resolution.

```
                ┌─────────────────────────┐
                │  Detection Coordinator  │
                │  ┌────────┐ ┌─────────┐ │
                │  │Pattern │ │  LLM    │ │
                │  │  (8x)  │ │(Ollama) │ │
                │  └───┬────┘ └───┬─────┘ │
                │      └────┬─────┘       │
                │           ▼             │
                │   Confidence Scoring    │
                └──────────┬──────────────┘
                           │
                    ┌──────┴──────┐
            conf<θ? │             │ conf≥θ?
                    ▼             ▼
            ┌─────────────┐ ┌─────────────────┐
            │ HITL Review │ │  Anonymization  │
            └──────┬──────┘ └────────┬────────┘
```

### 2. Weakness Analyzer
Iterative exploration loop that discovers failures and evolves prompts.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEAKNESS ANALYZER                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Benchmark → Identify Weaknesses → Hypothesize → Test → Root Cause      │
│     ▲                                                          │        │
│     └────────────────────── Loop ──────────────────────────────┘        │
│                                                                         │
│  Done? → Evolve Prompts → Report                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

For detailed component descriptions (Agents, Evolution, Curriculum), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
For a breakdown of agent roles, see [docs/AGENTS.md](docs/AGENTS.md).

### Weakness Taxonomy
Failures are categorized into types like `OVERLAP_CONFLICT`, `COVERAGE_GAP`, and `CONTEXT_DEPENDENCY`. See [docs/TAXONOMY.md](docs/TAXONOMY.md) for the full list.

## Project Structure

```
piigent/
├── agents/                 # Detection and self-critique agents
├── prompts/                # Prompt Evolution System
├── evaluation/             # Multi-Dimensional Evaluation
├── curriculum/             # Curriculum Learning
├── components/             # Core analysis components (parsers, resolvers)
├── graph/                  # LangGraph flow definitions
├── docs/                   # Detailed documentation & research notes
├── demo/                   # Runnable demo scripts
├── anoner/                 # Custom Presidio fork (submodule)
└── synpii/                 # Synthetic PII generator (submodule)
```

## Current Limitations

- **Recall Gap**: ~76-78% recall (core research problem)
- **Overlap Conflicts**: Pattern detections often overridden by LLM
- **Synthetic Data**: Validation currently relies on synthetic clinical text

**Do not use this for production GDPR/DSGVO compliance.** The purpose is to discover and document failure modes.

## References & Future Directions

For a detailed roadmap and research references (including ReAct, Reflexion, etc.), see [docs/REFERENCES.md](docs/REFERENCES.md).

## Author

[@shokrydev](https://github.com/shokrydev)
