# Agent Architecture Overview

 The `agents/` directory in this codebase contains a mix of autonomous LangGraph agents, deterministic nodes, and pure utility classes. They are not all "agents" in the autonomous sense, and not all use LLMs.

## 1. True LangGraph Agents (The "Brains")
 These components define a state, nodes, and a graph structure. They drive autonomous loops and make decisions.

 * **`weakness_analyzer.py`**: A full **LangGraph agent**.
     * **Role**: Autonomous loop for finding bugs.
     * **Mechanism**: Defines `WeaknessAnalyzerState`, has multiple nodes (benchmark, hypothesize, prompt_evolution), and uses conditional edges to loop until it solves a problem.
     * **LLM Usage**: Heavy (Ministral via Ollama) for hypothesis generation and reasoning.


## Agentic Concepts

PIIgent draws on patterns from recent agentic AI research:

| Concept | How PIIgent Uses It | Inspiration |
|---------|---------------------|-------------|
| **Reasoning + Acting** | Weakness Analyzer interleaves hypothesis generation with targeted test case creation | ReAct [Yao et al., 2022] |
| **Self-Critique** | Agents generate explanations for failures and propose fixes | Reflexion [Shinn et al., 2023] |
| **Ensemble Coordination** | Specialized recognizers (pattern-based, LLM-based) with orchestrated aggregation | Multi-agent systems [Han et al., 2024] |
| **Curriculum Learning** | Difficulty-aware progression with failure-weighted sampling | Bengio et al., 2009 |

## 2. LangGraph Nodes & Tools (The "Hands")
 These are designed to be *used* by LangGraph (often implementing `__call__(state)`), but they are typically single-step actions rather than autonomous loops.

 * **`detection_coordinator.py`**:
     * **Role**: Orchestrates the detection pipeline.
     * **Mechanism**: Acts as a deterministic plumbing component that aggregates results from multiple recognizers.
     * **LLM Usage**: Yes (calls Ministral via Ollama), but as a tool, not a decision-maker.
 * **`fix_proposal.py`**:
     * **Role**: Proposes code/prompt fixes.
     * **Mechanism**: Helper logic to translate error patterns into fix suggestions.
     * **LLM Usage**: Potential, designed to support LLM-based fix generation.
 * **`verification.py`**:
     * **Role**: Verifies fixes.
     * **Mechanism**: Runs benchmarks and tests. strictly an execution environment.
     * **LLM Usage**: No.

## 3. Pure Logic & Utilities (The "Tools")
 These are standard Python classes used by the agents. They are deterministic and do not use LLMs or LangGraph.

 * **`aggregator.py`**: Pure logic to merge overlapping PII detections (e.g., merging "John" and "John Doe").
 * **`synpii_integration.py`**: Wrapper around the synthetic data generator. Uses grammars and randomization.
 * **`weakness_to_mutation.py`**: Deterministic mapper that converts a "found weakness" into a "suggested prompt change."
 * **`error_taxonomy.py`**: Definitions for classifying errors (False Positives vs False Negatives).

## Summary Table

 | File | Is LangGraph Agent? | Uses LLM? | Role |
 | :--- | :--- | :--- | :--- |
 | **`weakness_analyzer.py`** | **Yes** (Full Graph) | **Yes** | Autonomous loop for weakness discovery. |
 | **`detection_coordinator.py`** | Node (Single Step) | **Yes** (Calls Model) | Runs detection tools. |
 | **`fix_proposal.py`** | Node/Helper | Maybe | Proposes fixes. |
 | **`verification.py`** | Node/Helper | No | Runs verification tests. |
 | **`aggregator.py`** | No (Utility) | No | Logic for entity merging. |
 | **`synpii_integration.py`** | No (Utility) | No | Generates synthetic data. |
 | **`weakness_to_mutation.py`** | No (Utility) | No | Maps errors to code changes. |
 | **`error_taxonomy.py`** | No (Utility) | No | Error classification logic. |
