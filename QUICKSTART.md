# Quick Start with PIIgent

## Installation

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
ollama pull ministral-3:8b
```

## Running the Pipeline

You can run the standard PII detection pipeline on a single document:

```bash
python .demo/run_agentic_flow.py
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

## Running Weakness Analysis

To explore where the pipeline fails and generate targeted test cases:

```bash
python .demo/run_weakness_analysis.py
```

Code example:
```python
from agents.weakness_analyzer import analyze_weaknesses

report = analyze_weaknesses(
    num_docs=20,
    max_rounds=3,
    enable_evolution=True,  # Automatically evolve prompts based on weaknesses
)

print(report["recommendations"])
print(report["evolution"]["mutations_applied"])  # Number of prompt mutations applied
print(report["evolution"]["evolved_genome_id"])  # ID of the improved genome
```
