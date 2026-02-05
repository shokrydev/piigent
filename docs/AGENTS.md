## 1. True LangGraph Agents (The "Brains")
 These components define a state, nodes, and a graph structure. They drive autonomous loops and make decisions.

 * **`agents/core/weakness_analyzer.py`**: A full **LangGraph agent**.
     * **Role**: Autonomous loop for finding bugs.
     * **Mechanism**: Defines `WeaknessAnalyzerState`, has multiple nodes (benchmark, hypothesize, prompt_evolution), and uses conditional edges to loop until it solves a problem.
     * **LLM Usage**: Heavy (Ministral via Ollama) for hypothesis generation and reasoning.

## 2. LangGraph Nodes & Tools (The "Hands")
 These are designed to be *used* by LangGraph (often implementing `__call__(state)`), but they are typically single-step actions rather than autonomous loops.

 * **`agents/core/detection_coordinator.py`**:
     * **Role**: Orchestrates the detection flow.
     * **Mechanism**: Aggregates results from various recognizers via `AnonerWrapper`.
     * **LLM Usage**: Yes (via wrapper).
 * **`agents/critics/fix_proposal.py`**:
     * **Role**: Proposes code/prompt fixes based on error analysis.
 * **`agents/critics/verification.py`**:
     * **Role**: Runs benchmarks and tests to verify fixes.

## 3. Pure Logic & Utilities (The "Tools")
 These are standard Python classes and wrappers used by the agents.

 * **`wrappers/anoner_wrapper.py`**:
     * **Role**: Centalized interface for the `anoner` (Presidio) fork.
     * **Capabilities**: Patterns, LLM (Ministral), and BERT/GLiNER models.
 * **`agents/helpers/aggregator.py`**: Pure logic to merge overlapping PII detections.
 * **`agents/helpers/synpii_integration.py`**: Wrapper around the synthetic data generator.
 * **`agents/core/weakness_to_mutation.py`**: Maps weaknesses to specific prompt mutations.
 * **`agents/critics/error_taxonomy.py`**: Classifies errors for the critics.

## Summary Table

 | File | Is LangGraph Agent? | Uses LLM? | Role |
 | :--- | :--- | :--- | :--- |
 | **`weakness_analyzer.py`** | **Yes** | **Yes** | Autonomous loop for weakness discovery. |
 | **`detection_coordinator.py`** | Node | **Yes** | Orchestrates detection tools. |
 | **`privacy_flow.py`** | **Flow** | **Yes** | Recursive privacy flow (Detect -> Anonymize -> Audit). |
 | **`anoner_wrapper.py`** | Wrapper | **Yes** | Isolated interface for Presidio fork. |
 | **`fix_proposal.py`** | Helper | Maybe | Proposes fixes. |
 | **`verification.py`** | Helper | No | Runs verification tests. |
 | **`aggregator.py`** | Utility | No | Logic for entity merging. |
