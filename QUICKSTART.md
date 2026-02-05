# Quick Start with PIIgent

## Installation

```bash
# Clone the repository
git clone https://github.com/shokrydev/piigent.git
cd piigent

# Sync the workspace (creates .venv and installs all dependencies)
uv sync

# Pull the required LLM for agentic detection
ollama pull ministral-3:8b
```

You can run the standard PII detection flow:

```bash
uv run demo/run_agentic_flow.py
```

Code example:
```python
from graph.privacy_flow import run_flow

result = run_flow(
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

To explore where the flow fails and generate targeted test cases:

```bash
uv run demo/run_weakness_analysis.py
```

Code example:
```python
from agents.core.weakness_analyzer import analyze_weaknesses

report = analyze_weaknesses(
    num_docs=20,
    max_rounds=3,
    enable_evolution=True,  # Automatically evolve prompts based on weaknesses
)

print(report["recommendations"])
print(report["evolution"]["mutations_applied"])  # Number of prompt mutations applied
print(report["evolution"]["evolved_genome_id"])  # ID of the improved genome
```
