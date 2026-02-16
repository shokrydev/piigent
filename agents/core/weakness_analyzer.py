"""Weakness Analyzer Agent - Iteratively explores anonymization weaknesses.

This agent implements an exploratory loop:
1. Run benchmark → Identify weak areas (low recall/precision)
2. Generate targeted synthetic data in weak areas
3. Run focused tests → Analyze failure patterns
4. Hypothesize root cause → Generate edge cases to confirm
5. Report findings with evidence

This is a LangGraph agent that uses tool calls to:
- Generate synthetic documents
- Run detection flow
- Analyze results
- Decide next exploration step
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated, Tuple
import operator

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)



DEFAULT_MODEL = "ministral-3:8b"


# =============================================================================
# Weakness Types
# =============================================================================

class WeaknessType(str, Enum):
    """Categories of anonymization weaknesses."""

    OVERLAP_CONFLICT = "overlap_conflict"
    """Higher-score entity overlaps and replaces correct detection.
    Example: LOCATION "53168 Bonn" (0.85) beats DE_POSTAL_CODE "53168" (0.30)
    """

    SCORE_THRESHOLD = "score_threshold"
    """Entity detected but score too low for confidence threshold.
    Example: DE_POSTAL_CODE at 0.30 when threshold is 0.70
    """

    CONTEXT_DEPENDENCY = "context_dependency"
    """Detection requires surrounding context that varies.
    Example: "Müller" detected as PERSON only when preceded by "Herr/Frau"
    """

    FORMAT_VARIATION = "format_variation"
    """Entity format varies from expected pattern.
    Example: KVNR with spaces "A 123 456 780" vs "A123456780"
    """

    ENTITY_CONFUSION = "entity_confusion"
    """Wrong entity type assigned.
    Example: BSNR detected as LANR (same format, different context)
    """

    COVERAGE_GAP = "coverage_gap"
    """Entity type not detected at all.
    Example: AGE patterns like "64-jährige" never recognized
    """

    LEAKAGE = "leakage"
    """PII survives anonymization.
    Example: "Herr Müller" not anonymized when "Hans Müller" was
    """


@dataclass
class TestCase:
    """A targeted test case for weakness exploration."""
    text: str
    expected_entities: Dict[str, List[str]]  # {entity_type: [text values]}
    description: str


@dataclass
class TestResult:
    """Result of running a test case."""
    test_case: TestCase
    detected_entities: List[Dict]  # [{entity_type, text, start, end, score}]
    expected_found: Dict[str, bool]  # {entity_type: was_detected}
    unexpected: List[Dict]  # Entities detected but not expected


@dataclass
class WeaknessEvidence:
    """Evidence supporting a weakness hypothesis."""
    test_input: str
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    explanation: str


@dataclass
class Weakness:
    """A discovered anonymization weakness."""
    weakness_type: WeaknessType
    entity_type: str
    description: str
    severity: float = 0.0  # 0.0-1.0
    evidence: List[WeaknessEvidence] = field(default_factory=list)
    root_cause: Optional[str] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "weakness_type": self.weakness_type.value,
            "entity_type": self.entity_type,
            "description": self.description,
            "severity": self.severity,
            "evidence": [asdict(e) for e in self.evidence],
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
        }


# =============================================================================
# Agent State
# =============================================================================

class ExplorationPhase(str, Enum):
    """Current phase of weakness exploration."""
    INITIAL_BENCHMARK = "initial_benchmark"
    WEAKNESS_IDENTIFICATION = "weakness_identification"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    TARGETED_TESTING = "targeted_testing"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PROMPT_EVOLUTION = "prompt_evolution"  # NEW: Apply prompt mutations
    REPORT_GENERATION = "report_generation"


class WeaknessAnalyzerState(TypedDict):
    """State for the Weakness Analyzer agent."""

    # Configuration
    num_initial_docs: int
    max_exploration_rounds: int
    confidence_threshold: float
    use_ministral: bool
    ministral_model: str
    preset: str

    # Exploration tracking
    current_phase: ExplorationPhase
    exploration_round: int

    # Benchmark results
    initial_metrics: Dict[str, Any]
    per_entity_metrics: Dict[str, Dict[str, float]]

    # Weakness tracking
    identified_weaknesses: List[Dict]
    current_weakness_index: int
    current_hypothesis: Optional[str]

    # Test data
    test_cases: List[Dict]
    test_results: List[Dict]

    # Agent reasoning
    messages: Annotated[List[Dict], operator.add]

    # Final output
    report: Optional[Dict]

    # Prompt evolution settings (NEW)
    enable_prompt_evolution: bool
    genome_store_path: str
    max_fixes_per_round: int

    # Evolution tracking (NEW)
    original_genome_id: Optional[str]
    evolved_genome_id: Optional[str]
    applied_mutations: List[Dict]
    evolution_metrics: Dict[str, float]


# =============================================================================
# Agent Nodes
# =============================================================================

def run_initial_benchmark(state: WeaknessAnalyzerState) -> Dict:
    """Run initial benchmark to establish baseline metrics."""
    from benchmarks.run_benchmark import run_benchmark
    from agents.helpers.synpii_integration import get_synpii_generator

    num_docs = state["num_initial_docs"]
    logger.info(f"Running initial benchmark with {num_docs} documents")

    # Generate documents using SynPII
    synpii_gen = get_synpii_generator(seed=42)
    doc_dicts = synpii_gen.generate_documents(num_docs)

    # Run benchmark
    metrics = run_benchmark(
        documents=doc_dicts,
        preset=state.get("preset", "clinical"),
        use_ministral=state.get("use_ministral", True),
        ministral_model=state.get("ministral_model", DEFAULT_MODEL),
        use_gliner=False,
        confidence_threshold=state.get("confidence_threshold", 0.7),
    )

    return {
        "initial_metrics": metrics["overall"],
        "per_entity_metrics": metrics["per_type"],
        "current_phase": ExplorationPhase.WEAKNESS_IDENTIFICATION,
        "messages": [{
            "role": "assistant",
            "content": f"Initial benchmark complete. Overall F1: {metrics['overall']['f1']:.2f}. "
                      f"Found {sum(1 for m in metrics['per_type'].values() if m['f1'] < 0.5)} "
                      f"entity types with F1 < 0.5"
        }]
    }


def identify_weaknesses(state: WeaknessAnalyzerState) -> Dict:
    """Analyze benchmark results to identify weaknesses."""
    weaknesses = []
    per_entity = state["per_entity_metrics"]

    for entity_type, metrics in per_entity.items():
        f1 = metrics.get("f1", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        support = metrics.get("support", 0)

        if support == 0:
            continue  # Skip types with no ground truth

        if f1 == 0:
            # Complete failure
            if entity_type == "DE_POSTAL_CODE":
                weakness = Weakness(
                    weakness_type=WeaknessType.OVERLAP_CONFLICT,
                    entity_type=entity_type,
                    description=f"{entity_type} likely overlapped by LOCATION (city adjacent to PLZ)",
                    severity=1.0,
                )
            elif entity_type == "AGE":
                weakness = Weakness(
                    weakness_type=WeaknessType.COVERAGE_GAP,
                    entity_type=entity_type,
                    description=f"{entity_type} patterns not recognized by any recognizer",
                    severity=0.8,
                )
            else:
                weakness = Weakness(
                    weakness_type=WeaknessType.COVERAGE_GAP,
                    entity_type=entity_type,
                    description=f"{entity_type} not detected (F1=0)",
                    severity=0.9,
                )
            weaknesses.append(weakness)

        elif recall < 0.7 and precision > recall:
            # Low recall suggests context dependency or format issues
            weakness = Weakness(
                weakness_type=WeaknessType.CONTEXT_DEPENDENCY,
                entity_type=entity_type,
                description=f"{entity_type} has low recall ({recall:.0%}) - likely context-dependent",
                severity=0.7 - recall,
            )
            weaknesses.append(weakness)

        elif precision < 0.7 and recall > precision:
            # Low precision suggests entity confusion
            weakness = Weakness(
                weakness_type=WeaknessType.ENTITY_CONFUSION,
                entity_type=entity_type,
                description=f"{entity_type} has low precision ({precision:.0%}) - possible type confusion",
                severity=0.7 - precision,
            )
            weaknesses.append(weakness)

    # Sort by severity
    weaknesses.sort(key=lambda w: w.severity, reverse=True)

    return {
        "identified_weaknesses": [w.to_dict() for w in weaknesses],
        "current_weakness_index": 0,
        "current_phase": ExplorationPhase.HYPOTHESIS_GENERATION if weaknesses else ExplorationPhase.REPORT_GENERATION,
        "messages": [{
            "role": "assistant",
            "content": f"Identified {len(weaknesses)} weaknesses. "
                      f"Top: {weaknesses[0].description if weaknesses else 'None'}"
        }]
    }


def generate_hypothesis(state: WeaknessAnalyzerState) -> Dict:
    """Generate hypothesis for the current weakness."""
    weaknesses = state["identified_weaknesses"]
    idx = state.get("current_weakness_index", 0)

    if idx >= len(weaknesses):
        return {
            "current_phase": ExplorationPhase.REPORT_GENERATION,
            "messages": [{"role": "assistant", "content": "All weaknesses analyzed."}]
        }

    weakness = weaknesses[idx]
    weakness_type = weakness["weakness_type"]
    entity_type = weakness["entity_type"]

    hypotheses = {
        WeaknessType.OVERLAP_CONFLICT.value: (
            f"HYPOTHESIS: {entity_type} is detected by pattern recognizer but a higher-scoring "
            f"LLM detection (likely LOCATION) overlaps and replaces it in aggregation. "
            f"TEST: Generate cases with varying entity arrangements to confirm overlap behavior."
        ),
        WeaknessType.COVERAGE_GAP.value: (
            f"HYPOTHESIS: No recognizer (pattern or LLM) outputs {entity_type}. "
            f"TEST: Generate isolated {entity_type} instances to confirm zero detection rate."
        ),
        WeaknessType.CONTEXT_DEPENDENCY.value: (
            f"HYPOTHESIS: {entity_type} detection depends on surrounding context words. "
            f"TEST: Generate same entity with varying context to identify triggers."
        ),
        WeaknessType.FORMAT_VARIATION.value: (
            f"HYPOTHESIS: {entity_type} format varies from expected pattern. "
            f"TEST: Generate format variations (spaces, dashes, case) to find failures."
        ),
        WeaknessType.ENTITY_CONFUSION.value: (
            f"HYPOTHESIS: {entity_type} is being confused with another entity type. "
            f"TEST: Generate unambiguous {entity_type} instances to check type assignment."
        ),
    }

    hypothesis = hypotheses.get(weakness_type, f"Investigating {entity_type} {weakness_type}")

    return {
        "current_hypothesis": hypothesis,
        "current_phase": ExplorationPhase.TARGETED_TESTING,
        "messages": [{"role": "assistant", "content": hypothesis}]
    }


def run_targeted_tests(state: WeaknessAnalyzerState) -> Dict:
    """Generate and run targeted test cases."""
    from graph.privacy_flow import run_flow
    from agents.helpers.synpii_integration import get_synpii_generator

    weaknesses = state["identified_weaknesses"]
    idx = state.get("current_weakness_index", 0)

    if idx >= len(weaknesses):
        return {"current_phase": ExplorationPhase.REPORT_GENERATION}

    weakness = weaknesses[idx]

    # Generate test cases using SynPII
    generator = get_synpii_generator(seed=42 + idx)
    test_cases = generator.generate_for_weakness(weakness, count=10)

    if not test_cases:
        logger.warning(f"No test cases generated for {weakness['entity_type']}")
        return {
            "test_cases": [],
            "test_results": [],
            "current_phase": ExplorationPhase.ROOT_CAUSE_ANALYSIS,
        }

    # Run tests
    test_results = []
    for tc in test_cases:
        try:
            result = run_flow(
                document=tc.text,
                confidence_threshold=state.get("confidence_threshold", 0.7),
                human_in_loop=False,
                preset=state.get("preset", "clinical"),
                use_ministral=state.get("use_ministral", True),
                ministral_model=state.get("ministral_model", DEFAULT_MODEL),
            )

            detected = result.get("detected_entities", [])

            # Analyze results
            detected_dicts = [
                {"entity_type": e.entity_type, "text": e.text, "start": e.start, "end": e.end, "score": e.score}
                for e in detected
            ]

            expected_found = {}
            for exp_type, exp_values in tc.expected_entities.items():
                found = any(
                    d["entity_type"] == exp_type and any(v in d["text"] or d["text"] in v for v in exp_values)
                    for d in detected_dicts
                )
                expected_found[exp_type] = found

            unexpected = [
                d for d in detected_dicts
                if d["entity_type"] not in tc.expected_entities
            ]

            test_results.append({
                "test_case": {"text": tc.text, "expected": tc.expected_entities, "description": tc.description},
                "detected": detected_dicts,
                "expected_found": expected_found,
                "unexpected": unexpected,
            })

        except Exception as e:
            logger.error(f"Test failed for '{tc.text[:50]}...': {e}")
            test_results.append({
                "test_case": {"text": tc.text, "expected": tc.expected_entities, "description": tc.description},
                "error": str(e),
            })

    return {
        "test_cases": [{"text": tc.text, "expected": tc.expected_entities, "description": tc.description} for tc in test_cases],
        "test_results": test_results,
        "current_phase": ExplorationPhase.ROOT_CAUSE_ANALYSIS,
        "exploration_round": state.get("exploration_round", 0) + 1,
        "messages": [{
            "role": "assistant",
            "content": f"Ran {len(test_cases)} targeted tests for {weakness['entity_type']}"
        }]
    }


def analyze_root_cause(state: WeaknessAnalyzerState) -> Dict:
    """Analyze test results to determine root cause."""
    weaknesses = state["identified_weaknesses"]
    idx = state.get("current_weakness_index", 0)
    test_results = state.get("test_results", [])

    if idx >= len(weaknesses):
        return {"current_phase": ExplorationPhase.REPORT_GENERATION}

    weakness = weaknesses[idx]
    weakness_type = weakness["weakness_type"]
    entity_type = weakness["entity_type"]

    # Analyze patterns in test results
    total_tests = len(test_results)
    expected_found_count = sum(
        1 for r in test_results
        if r.get("expected_found", {}).get(entity_type, False)
    )
    overlap_detected = sum(
        1 for r in test_results
        if any(u.get("entity_type") == "LOCATION" for u in r.get("unexpected", []))
    )

    # Generate root cause and fix based on analysis
    root_cause = None
    suggested_fix = None
    evidence = []

    if weakness_type == WeaknessType.OVERLAP_CONFLICT.value:
        if overlap_detected > 0:
            root_cause = (
                f"Confirmed: In {overlap_detected}/{total_tests} tests, LOCATION was detected "
                f"overlapping with {entity_type}. The aggregation strategy uses max-score, "
                f"so LLM's LOCATION (typically 0.85) beats pattern's {entity_type} (typically 0.30)."
            )
            suggested_fix = (
                "Options: (1) Prefer pattern recognizers for structured German IDs in aggregation, "
                "(2) Split compound LOCATION detections into components, "
                "(3) Increase pattern recognizer base scores for German healthcare entities."
            )
        else:
            root_cause = f"{entity_type} not detected even in isolation - may be pattern issue."
            suggested_fix = f"Check {entity_type} pattern recognizer configuration."

    elif weakness_type == WeaknessType.COVERAGE_GAP.value:
        if expected_found_count == 0:
            root_cause = (
                f"Confirmed: 0/{total_tests} tests detected {entity_type}. "
                f"No recognizer (pattern or LLM) is configured to output this entity type."
            )
            suggested_fix = (
                f"Options: (1) Add a pattern recognizer for {entity_type}, "
                f"(2) Update LLM prompt to extract {entity_type}, "
                f"(3) Add {entity_type} to GLiNER entity list."
            )
        else:
            root_cause = f"Partial detection: {expected_found_count}/{total_tests} - format or context issue."

    elif weakness_type == WeaknessType.CONTEXT_DEPENDENCY.value:
        # Analyze which contexts worked vs failed
        working_contexts = [r for r in test_results if r.get("expected_found", {}).get(entity_type)]
        failing_contexts = [r for r in test_results if not r.get("expected_found", {}).get(entity_type)]

        if working_contexts and failing_contexts:
            root_cause = (
                f"Context-dependent: {len(working_contexts)}/{total_tests} detected. "
                f"Working: {[r['test_case']['description'] for r in working_contexts[:2]]}. "
                f"Failing: {[r['test_case']['description'] for r in failing_contexts[:2]]}."
            )
            suggested_fix = "Add context words to recognizer or use LLM for better contextual detection."

    # Build evidence from test results
    for r in test_results[:3]:
        if "error" not in r:
            evidence.append(WeaknessEvidence(
                test_input=r["test_case"]["text"],
                expected=r["test_case"]["expected"],
                actual={"detected": r.get("detected", [])},
                explanation=f"Expected {entity_type}: {r.get('expected_found', {}).get(entity_type, 'N/A')}"
            ))

    # Update weakness with findings
    updated_weakness = weakness.copy()
    updated_weakness["root_cause"] = root_cause
    updated_weakness["suggested_fix"] = suggested_fix
    updated_weakness["evidence"] = [asdict(e) for e in evidence]

    updated_weaknesses = state["identified_weaknesses"].copy()
    updated_weaknesses[idx] = updated_weakness

    # Decide next step
    next_idx = idx + 1
    max_rounds = state.get("max_exploration_rounds", 3)
    current_round = state.get("exploration_round", 1)

    if next_idx < len(weaknesses) and current_round < max_rounds:
        next_phase = ExplorationPhase.HYPOTHESIS_GENERATION
    else:
        next_phase = ExplorationPhase.REPORT_GENERATION

    return {
        "identified_weaknesses": updated_weaknesses,
        "current_weakness_index": next_idx,
        "current_phase": next_phase,
        "messages": [{
            "role": "assistant",
            "content": f"Root cause for {entity_type}: {root_cause[:100] if root_cause else 'Unknown'}..."
        }]
    }


def generate_report(state: WeaknessAnalyzerState) -> Dict:
    """Generate final weakness analysis report."""
    weaknesses = state["identified_weaknesses"]

    report = {
        "summary": {
            "initial_f1": state["initial_metrics"].get("f1", 0),
            "initial_precision": state["initial_metrics"].get("precision", 0),
            "initial_recall": state["initial_metrics"].get("recall", 0),
            "weaknesses_found": len(weaknesses),
            "weaknesses_analyzed": state.get("current_weakness_index", 0),
            "exploration_rounds": state.get("exploration_round", 0),
        },
        "weaknesses": weaknesses,
        "recommendations": [],
        "per_entity_metrics": state["per_entity_metrics"],
        # NEW: Evolution section
        "evolution": {
            "enabled": state.get("enable_prompt_evolution", True),
            "original_genome_id": state.get("original_genome_id"),
            "evolved_genome_id": state.get("evolved_genome_id"),
            "mutations_applied": len(state.get("applied_mutations", [])),
            "applied_mutations": state.get("applied_mutations", []),
            "metrics": state.get("evolution_metrics", {}),
        },
    }

    # Compile recommendations (prioritized)
    for w in weaknesses:
        if w.get("suggested_fix"):
            report["recommendations"].append({
                "priority": 1 if w["severity"] > 0.8 else 2 if w["severity"] > 0.5 else 3,
                "entity_type": w["entity_type"],
                "weakness_type": w["weakness_type"],
                "fix": w["suggested_fix"],
                "root_cause": w.get("root_cause"),
            })

    # Sort by priority
    report["recommendations"].sort(key=lambda x: x["priority"])

    return {
        "report": report,
        "current_phase": ExplorationPhase.REPORT_GENERATION,
        "messages": [{
            "role": "assistant",
            "content": f"Analysis complete. Found {len(weaknesses)} weaknesses. "
                      f"Generated {len(report['recommendations'])} recommendations. "
                      f"Evolved genome: {state.get('evolved_genome_id', 'N/A')}"
        }]
    }


def run_prompt_evolution(state: WeaknessAnalyzerState) -> Dict:
    """Apply prompt evolution based on discovered weaknesses.
    
    Maps weaknesses to mutations, applies them via MutationEngine,
    verifies improvements with VerificationAgent, and stores
    successful genomes.
    """
    from agents.core.weakness_to_mutation import WeaknessToMutationMapper
    from agents.critics.fix_proposal import FixProposalAgent
    from agents.critics.verification import VerificationAgent, TestCase as VerificationTestCase
    from prompts.store import PromptGenomeStore
    from prompts.mutations import MutationEngine, create_mutation_from_fix
    
    # Check if evolution is enabled
    if not state.get("enable_prompt_evolution", True):
        return {
            "current_phase": ExplorationPhase.REPORT_GENERATION,
            "messages": [{"role": "assistant", "content": "Prompt evolution disabled, skipping."}]
        }
    
    weaknesses = state.get("identified_weaknesses", [])
    if not weaknesses:
        return {
            "current_phase": ExplorationPhase.REPORT_GENERATION,
            "messages": [{"role": "assistant", "content": "No weaknesses to evolve prompts for."}]
        }
    
    # Initialize components
    store_path = state.get("genome_store_path", "prompt_genomes.db")
    max_fixes = state.get("max_fixes_per_round", 3)
    
    try:
        store = PromptGenomeStore(store_path)
        mapper = WeaknessToMutationMapper(max_mutations_per_weakness=2)
        mutation_engine = MutationEngine()
        verification_agent = VerificationAgent()
        fix_proposal_agent = FixProposalAgent()
        
        # Load or create base genome
        genome = store.get_best() or store.ensure_default_exists()
        original_genome_id = genome.id
        
        logger.info(f"Starting evolution from genome: {original_genome_id}")
        
        # Map weaknesses to mutations
        all_proposals = mapper.map_weaknesses_batch(weaknesses, max_total_mutations=max_fixes * 2)
        
        # Also generate proposals from evidence via FixProposalAgent
        for weakness in weaknesses[:3]:  # Top 3 weaknesses
            evidence = weakness.get("evidence", [])
            if evidence:
                errors = mapper.convert_evidence_to_errors(evidence, weakness["entity_type"])
                if errors:
                    fix_proposals = fix_proposal_agent.propose_fixes(errors, genome)
                    all_proposals.extend(fix_proposals[:2])
        
        # Sort by priority and take top N
        all_proposals.sort(key=lambda p: (p.priority, -p.estimated_impact))
        proposals_to_try = all_proposals[:max_fixes]
        
        # Convert test cases from state to verification format
        test_cases = []
        for tc_dict in state.get("test_cases", []):
            test_cases.append(VerificationTestCase(
                text=tc_dict.get("text", ""),
                expected_entities=[
                    {"entity_type": et, "text": val}
                    for et, vals in tc_dict.get("expected", {}).items()
                    for val in (vals if isinstance(vals, list) else [vals])
                ],
            ))
        
        # Apply mutations cumulatively, keeping only improvements
        applied_mutations = []
        current_genome = genome
        
        for proposal in proposals_to_try:
            try:
                # Apply mutation
                operator, params = create_mutation_from_fix(proposal)
                mutated_genome = mutation_engine.apply_mutation(
                    current_genome, operator, params
                )
                
                # Verify if we have test cases
                if test_cases:
                    result = verification_agent.verify_fix(
                        original_prompt=current_genome,
                        fixed_prompt=mutated_genome,
                        test_cases=test_cases,
                    )
                    
                    if result.improved and not result.regression_detected:
                        # Improvement confirmed
                        current_genome = mutated_genome
                        applied_mutations.append({
                            "mutation": proposal.mutation.value,
                            "target": proposal.target,
                            "rationale": proposal.rationale,
                            "delta_f1": result.delta_f1,
                        })
                        logger.info(f"Applied mutation: {proposal.mutation.value} for {proposal.target}")
                else:
                    # No test cases, apply mutation optimistically
                    current_genome = mutated_genome
                    applied_mutations.append({
                        "mutation": proposal.mutation.value,
                        "target": proposal.target,
                        "rationale": proposal.rationale,
                        "delta_f1": 0.0,  # Unknown without verification
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to apply mutation {proposal.mutation.value}: {e}")
                continue
        
        # Save evolved genome if we made changes
        evolved_genome_id = None
        evolution_metrics = {}
        
        if applied_mutations:
            store.save(current_genome)
            evolved_genome_id = current_genome.id
            
            evolution_metrics = {
                "mutations_attempted": len(proposals_to_try),
                "mutations_applied": len(applied_mutations),
                "total_delta_f1": sum(m.get("delta_f1", 0) for m in applied_mutations),
            }
            
            logger.info(f"Evolved genome saved: {evolved_genome_id}")
        
        return {
            "current_phase": ExplorationPhase.REPORT_GENERATION,
            "original_genome_id": original_genome_id,
            "evolved_genome_id": evolved_genome_id,
            "applied_mutations": applied_mutations,
            "evolution_metrics": evolution_metrics,
            "messages": [{
                "role": "assistant",
                "content": f"Prompt evolution complete. Applied {len(applied_mutations)} mutations. "
                          f"Evolved genome: {evolved_genome_id or 'No improvements found'}"
            }]
        }
        
    except Exception as e:
        logger.error(f"Prompt evolution failed: {e}")
        return {
            "current_phase": ExplorationPhase.REPORT_GENERATION,
            "applied_mutations": [],
            "evolution_metrics": {"error": str(e)},
            "messages": [{"role": "assistant", "content": f"Prompt evolution failed: {e}"}]
        }


# =============================================================================
# Graph Construction
# =============================================================================

def route_after_root_cause(state: WeaknessAnalyzerState) -> Literal["hypothesis", "evolution"]:
    """Decide whether to continue exploring or proceed to evolution."""
    phase = state.get("current_phase")

    if phase == ExplorationPhase.REPORT_GENERATION:
        return "evolution"  # Go through evolution before report
    if phase == ExplorationPhase.PROMPT_EVOLUTION:
        return "evolution"
    if phase == ExplorationPhase.HYPOTHESIS_GENERATION:
        return "hypothesis"
    return "evolution"  # Default to evolution


def create_weakness_analyzer_graph() -> StateGraph:
    """Create the Weakness Analyzer agent graph."""
    graph = StateGraph(WeaknessAnalyzerState)

    # Add nodes
    graph.add_node("initial_benchmark", run_initial_benchmark)
    graph.add_node("identify_weaknesses", identify_weaknesses)
    graph.add_node("generate_hypothesis", generate_hypothesis)
    graph.add_node("targeted_tests", run_targeted_tests)
    graph.add_node("root_cause_analysis", analyze_root_cause)
    graph.add_node("prompt_evolution", run_prompt_evolution)  # NEW node
    graph.add_node("generate_report", generate_report)

    # Add edges
    graph.set_entry_point("initial_benchmark")
    graph.add_edge("initial_benchmark", "identify_weaknesses")
    graph.add_edge("identify_weaknesses", "generate_hypothesis")
    graph.add_edge("generate_hypothesis", "targeted_tests")
    graph.add_edge("targeted_tests", "root_cause_analysis")

    # Conditional edge after root cause analysis
    graph.add_conditional_edges(
        "root_cause_analysis",
        route_after_root_cause,
        {
            "hypothesis": "generate_hypothesis",
            "evolution": "prompt_evolution",  # Route to evolution instead of report
        }
    )

    # Evolution leads to report
    graph.add_edge("prompt_evolution", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# =============================================================================
# Public API
# =============================================================================

def analyze_weaknesses(
    num_docs: int = 20,
    max_rounds: int = 3,
    confidence_threshold: float = 0.7,
    use_ministral: bool = True,
    ministral_model: str = "ministral-3:8b",
    preset: str = "clinical",
    # NEW: Prompt evolution parameters
    enable_evolution: bool = True,
    genome_store_path: str = "prompt_genomes.db",
    max_fixes_per_round: int = 3,
) -> Dict:
    """Run weakness analysis on the anonymization flow.

    Uses SynPII for grammar-aware synthetic document and test case generation.

    Args:
        num_docs: Number of documents for initial benchmark.
        max_rounds: Maximum exploration rounds.
        confidence_threshold: Flow confidence threshold.
        use_ministral: Enable Ministral LLM recognizer.
        ministral_model: Ollama model name.
        preset: Recognizer preset.
        enable_evolution: Enable automatic prompt evolution based on weaknesses.
        genome_store_path: Path to the genome store database.
        max_fixes_per_round: Maximum mutations to attempt per round.

    Returns:
        Weakness analysis report with evolution section.
    """
    graph = create_weakness_analyzer_graph()

    initial_state: WeaknessAnalyzerState = {
        "num_initial_docs": num_docs,
        "max_exploration_rounds": max_rounds,
        "confidence_threshold": confidence_threshold,
        "use_ministral": use_ministral,
        "ministral_model": ministral_model,
        "preset": preset,
        "current_phase": ExplorationPhase.INITIAL_BENCHMARK,
        "exploration_round": 0,
        "initial_metrics": {},
        "per_entity_metrics": {},
        "identified_weaknesses": [],
        "current_weakness_index": 0,
        "current_hypothesis": None,
        "test_cases": [],
        "test_results": [],
        "messages": [],
        "report": None,
        # NEW: Evolution settings
        "enable_prompt_evolution": enable_evolution,
        "genome_store_path": genome_store_path,
        "max_fixes_per_round": max_fixes_per_round,
        "original_genome_id": None,
        "evolved_genome_id": None,
        "applied_mutations": [],
        "evolution_metrics": {},
    }

    result = graph.invoke(initial_state)
    return result["report"]


def print_report(report: Dict) -> None:
    """Pretty-print a weakness analysis report."""
    print("\n" + "=" * 70)
    print("WEAKNESS ANALYSIS REPORT")
    print("=" * 70)

    summary = report["summary"]
    print(f"\nInitial Benchmark: P={summary['initial_precision']:.2f}, "
          f"R={summary['initial_recall']:.2f}, F1={summary['initial_f1']:.2f}")
    print(f"Weaknesses Found: {summary['weaknesses_found']}")
    print(f"Exploration Rounds: {summary['exploration_rounds']}")

    print("\n" + "-" * 70)
    print("IDENTIFIED WEAKNESSES")
    print("-" * 70)

    for i, w in enumerate(report["weaknesses"], 1):
        print(f"\n{i}. [{w['weakness_type'].upper()}] {w['entity_type']}")
        print(f"   Severity: {w['severity']:.1f}")
        print(f"   Description: {w['description']}")
        if w.get("root_cause"):
            print(f"   Root Cause: {w['root_cause'][:100]}...")
        if w.get("suggested_fix"):
            print(f"   Suggested Fix: {w['suggested_fix'][:100]}...")

    print("\n" + "-" * 70)
    print("RECOMMENDATIONS (Priority Order)")
    print("-" * 70)

    for i, rec in enumerate(report["recommendations"], 1):
        print(f"\n{i}. [P{rec['priority']}] {rec['entity_type']}: {rec['fix'][:80]}...")

    # NEW: Evolution section
    if "evolution" in report:
        print("\n" + "-" * 70)
        print("PROMPT EVOLUTION")
        print("-" * 70)
        
        evolution = report["evolution"]
        print(f"\nEnabled: {evolution.get('enabled', True)}")
        print(f"Original Genome: {evolution.get('original_genome_id', 'N/A')}")
        print(f"Evolved Genome: {evolution.get('evolved_genome_id', 'N/A')}")
        print(f"Mutations Applied: {evolution.get('mutations_applied', 0)}")
        
        if evolution.get("applied_mutations"):
            print("\nApplied Mutations:")
            for i, m in enumerate(evolution["applied_mutations"], 1):
                print(f"  {i}. [{m['mutation']}] {m['target']}: {m.get('rationale', '')[:50]}...")
        
        if "metrics" in evolution and evolution["metrics"]:
            metrics = evolution["metrics"]
            if "error" not in metrics:
                print(f"\nMetrics:")
                print(f"  Mutations Attempted: {metrics.get('mutations_attempted', 0)}")
                print(f"  Mutations Applied: {metrics.get('mutations_applied', 0)}")
                print(f"  Total Delta F1: {metrics.get('total_delta_f1', 0):.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    print("Running weakness analysis...", file=sys.stderr)
    report = analyze_weaknesses(
        num_docs=10,
        max_rounds=2,
        use_ministral=True,
        ministral_model="ministral-3:8b",
    )

    print_report(report)

    # Also output JSON for programmatic use
    print("\n\nJSON Report:")
    print(json.dumps(report, indent=2, default=str))
