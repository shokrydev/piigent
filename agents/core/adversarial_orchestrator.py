"""Red-Blue Orchestrator - Autonomous Adversarial Safety Agent.

This agent implements a Red-Blue Teaming loop to guarantee PII anonymization robustness:
1. **Red Team (Attack)**: Proactively generates adversarial scenarios (SynPII) to find weaknesses.
2. **Blue Team (Defense)**: Analyzes failures and evolves the system (Prompt Engineering).
3. **Orchestrator**: Manages the loop until robust.

Architecture:
- Attack Phase: Vulnerability Scan -> Attack Planning -> Red Team Execution
- Defense Phase: Root Cause Analysis -> Blue Team Optimization -> Verification
"""

import json
import logging
import operator
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from langgraph.graph import StateGraph

# Import shared types from SynPII to ensure Glossary alignment
from synpii.adversarial.types import AdversarialType

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ministral-3:8b"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AdversarialFinding:
    """A confirmed robustness failure (Vulnerability)."""
    failure_mode: AdversarialType
    entity_type: str
    description: str
    severity: float = 0.0  # 0.0-1.0
    evidence: List[Dict] = field(default_factory=list)
    root_cause: Optional[str] = None
    defense_strategy: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "failure_mode": self.failure_mode.value,
            "entity_type": self.entity_type,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
            "root_cause": self.root_cause,
            "defense_strategy": self.defense_strategy,
        }


# =============================================================================
# Agent State
# =============================================================================

class OrchestratorPhase(str, Enum):
    """Phases of the Red-Blue loop."""
    BASELINE = "baseline"
    VULNERABILITY_SCAN = "vulnerability_scan"  # Initial identification
    ATTACK_PLANNING = "attack_planning"        # Hypothesis generation
    RED_TEAM_EXECUTION = "red_team_execution"  # Targeted testing
    BLUE_TEAM_ANALYSIS = "blue_team_analysis"  # Root cause
    BLUE_TEAM_OPTIMIZATION = "blue_team_optimization" # Prompt evolution
    REPORTING = "reporting"


class RedBlueState(TypedDict):
    """State for the Red-Blue Orchestrator."""

    # Configuration
    num_initial_docs: int
    max_rounds: int
    confidence_threshold: float
    use_ministral: bool
    ministral_model: str
    preset: str
    
    # Execution Tracking
    current_phase: OrchestratorPhase
    current_round: int
    
    # Metrics
    initial_metrics: Dict[str, Any]
    per_entity_metrics: Dict[str, Dict[str, float]]
    
    # Adversarial Data
    findings: List[Dict]  # List of AdversarialFinding dicts
    current_finding_index: int
    current_attack_plan: Optional[str]
    
    # Test Data (The "Battlefield")
    attack_cases: List[Dict]
    battle_results: List[Dict]
    
    # Blue Team State
    enable_optimization: bool
    genome_store_path: str
    max_fixes_per_round: int
    applied_defenses: List[Dict]
    optimization_metrics: Dict[str, float]
    
    # Agent Communication
    messages: Annotated[List[Dict], operator.add]
    
    # Final Output
    report: Optional[Dict]
    final_genome_id: Optional[str]


# =============================================================================
# Agent Nodes
# =============================================================================

def run_baseline(state: RedBlueState) -> Dict:
    """Establish baseline performance (Passive Scan)."""
    from benchmarks.run_benchmark import run_benchmark
    from agents.helpers.synpii_integration import get_synpii_generator

    num_docs = state["num_initial_docs"]
    logger.info(f"Running baseline benchmark with {num_docs} documents")

    # Generate baseline traffic
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
        "current_phase": OrchestratorPhase.VULNERABILITY_SCAN,
        "messages": [{
            "role": "assistant",
            "content": f"Baseline established. Overall F1: {metrics['overall']['f1']:.2f}."
        }]
    }


def scan_for_vulnerabilities(state: RedBlueState) -> Dict:
    """Analyze baseline for potential vulnerabilities."""
    findings = []
    per_entity = state["per_entity_metrics"]

    for entity_type, metrics in per_entity.items():
        f1 = metrics.get("f1", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        support = metrics.get("support", 0)

        if support == 0:
            continue

        if f1 == 0:
            # Critical Failure
            if entity_type == "DE_POSTAL_CODE":
                finding = AdversarialFinding(
                    failure_mode=AdversarialType.OVERLAP_CONFLICT,
                    entity_type=entity_type,
                    description=f"Likely overlapped by larger entity (LOCATION)",
                    severity=1.0,
                )
            else:
                finding = AdversarialFinding(
                    failure_mode=AdversarialType.COVERAGE_GAP,
                    entity_type=entity_type,
                    description="Complete detection failure (F1=0)",
                    severity=0.9,
                )
            findings.append(finding)

        elif recall < 0.7 and precision > recall:
            # Context Failure
            finding = AdversarialFinding(
                failure_mode=AdversarialType.CONTEXT_DEPENDENCY,
                entity_type=entity_type,
                description=f"Low recall ({recall:.0%}) indicates context dependency",
                severity=0.7 - recall,
            )
            findings.append(finding)

        elif precision < 0.7 and recall > precision:
            # Confusion Failure
            finding = AdversarialFinding(
                failure_mode=AdversarialType.ENTITY_CONFUSION,
                entity_type=entity_type,
                description=f"Low precision ({precision:.0%}) indicates type confusion",
                severity=0.7 - precision,
            )
            findings.append(finding)

    findings.sort(key=lambda x: x.severity, reverse=True)

    return {
        "findings": [f.to_dict() for f in findings],
        "current_finding_index": 0,
        "current_phase": OrchestratorPhase.ATTACK_PLANNING if findings else OrchestratorPhase.REPORTING,
        "messages": [{
            "role": "assistant",
            "content": f"Scan found {len(findings)} potential vulnerabilities."
        }]
    }


def plan_attack(state: RedBlueState) -> Dict:
    """Red Team: Formulate an attack plan for the current vulnerability."""
    findings = state["findings"]
    idx = state.get("current_finding_index", 0)

    if idx >= len(findings):
        return {
            "current_phase": OrchestratorPhase.REPORTING,
            "messages": [{"role": "assistant", "content": "All vulnerabilities engaged."}]
        }

    finding = findings[idx]
    failure_mode = finding["failure_mode"]
    entity_type = finding["entity_type"]

    # Red Team Strategy
    strategies = {
        AdversarialType.OVERLAP_CONFLICT.value: 
            f"Generate compound entities where {entity_type} is embedded in competing types.",
        AdversarialType.COVERAGE_GAP.value: 
            f"Isolate {entity_type} in clean contexts to test fundamental recognition.",
        AdversarialType.CONTEXT_DEPENDENCY.value: 
            f"Vary context window around {entity_type} to identify blind spots.",
        AdversarialType.FORMAT_VARIATION.value: 
            f"Fuzz format of {entity_type} (separators, casing, spacing).",
        AdversarialType.ENTITY_CONFUSION.value: 
            f"Generate ambiguous examples to force misclassification of {entity_type}.",
    }
    
    plan = strategies.get(failure_mode, f"Stress test {entity_type} for {failure_mode}")

    return {
        "current_attack_plan": plan,
        "current_phase": OrchestratorPhase.RED_TEAM_EXECUTION,
        "messages": [{"role": "assistant", "content": f"Red Team Plan: {plan}"}]
    }


def execute_red_team(state: RedBlueState) -> Dict:
    """Red Team: Execute the attack using SynPII."""
    from graph.privacy_flow import run_flow
    from agents.helpers.synpii_integration import get_synpii_generator

    findings = state["findings"]
    idx = state.get("current_finding_index", 0)
    finding = findings[idx]

    # Generate Attack Vectors
    generator = get_synpii_generator(seed=42 + idx)
    # Note: synpii_integration needs to be updated to handle new AdversarialType enum values if generic strings used
    # But since we use values compatible with the wrapper, it should work.
    attack_cases = generator.generate_for_weakness(finding, count=10)

    if not attack_cases:
        return {
            "attack_cases": [],
            "battle_results": [],
            "current_phase": OrchestratorPhase.BLUE_TEAM_ANALYSIS,
        }

    # Execute Attack
    battle_results = []
    for case in attack_cases:
        try:
            result = run_flow(
                document=case.text,
                confidence_threshold=state.get("confidence_threshold", 0.7),
                human_in_loop=False,
                preset=state.get("preset", "clinical"),
                use_ministral=state.get("use_ministral", True),
                ministral_model=state.get("ministral_model", DEFAULT_MODEL),
            )
            
            # Record battle outcome
            detected = result.get("detected_entities", [])
            detected_dicts = [
                {"entity_type": e.entity_type, "text": e.text, "start": e.start, "end": e.end}
                for e in detected
            ]
            
            # Simple hit calculation
            expected_map = case.expected_entities
            hits = {}
            for et, values in expected_map.items():
                hit = any(d["entity_type"] == et and any(v in d["text"] for v in values) for d in detected_dicts)
                hits[et] = hit
            
            battle_results.append({
                "input": case.text,
                "expected": expected_map,
                "detected": detected_dicts,
                "hits": hits
            })

        except Exception as e:
            logger.error(f"Attack execution failed: {e}")

    return {
        "attack_cases": [{"text": c.text, "expected": c.expected_entities} for c in attack_cases],
        "battle_results": battle_results,
        "current_phase": OrchestratorPhase.BLUE_TEAM_ANALYSIS,
        "current_round": state.get("current_round", 1) + 1,
        "messages": [{"role": "assistant", "content": f"Red Team executed {len(attack_cases)} attacks."}]
    }


def analyze_root_cause(state: RedBlueState) -> Dict:
    """Blue Team: Analyze attack results to verify vulnerability."""
    findings = state["findings"]
    idx = state.get("current_finding_index", 0)
    results = state.get("battle_results", [])
    
    if idx >= len(findings):
        return {"current_phase": OrchestratorPhase.REPORTING} # Should not happen

    finding = findings[idx]
    entity_type = finding["entity_type"]
    
    # Calculate Defense Rate
    total_battles = len(results)
    if total_battles == 0:
        defense_rate = 1.0 # No attacks = Success? Or nothing to analyze.
    else:
        successful_defenses = sum(1 for r in results if r["hits"].get(entity_type, False))
        defense_rate = successful_defenses / total_battles

    root_cause = f"Defense Rate: {defense_rate:.1%}. "
    if defense_rate < 0.8:
        root_cause += "System failed to withstand Red Team attack. "
        if finding["failure_mode"] == AdversarialType.OVERLAP_CONFLICT.value:
             root_cause += "Aggregator logic favored overlapping entities."
        elif finding["failure_mode"] == AdversarialType.COVERAGE_GAP.value:
             root_cause += "Lack of recognition rules/knowledge."
    else:
        root_cause += "System held up against specific attack vector."

    # Update finding
    updated_finding = finding.copy()
    updated_finding["root_cause"] = root_cause
    updated_finding["severity"] = (1.0 - defense_rate) # Dynamically update severity based on attack success
    
    # Update evidence
    evidence = []
    for r in results[:3]:
        evidence.append({
            "test_input": r["input"],
            "expected": r["expected"],
            "actual": {"detected": r["detected"]},
            "explanation": f"Hit: {r['hits']}"
        })
    updated_finding["evidence"] = evidence
    
    updated_findings = state["findings"].copy()
    updated_findings[idx] = updated_finding

    # Decide next step
    next_phase = OrchestratorPhase.BLUE_TEAM_OPTIMIZATION
    
    return {
        "findings": updated_findings,
        "current_phase": next_phase,
        "messages": [{"role": "assistant", "content": f"Blue Team Analysis: {root_cause}"}]
    }


def execute_blue_team(state: RedBlueState) -> Dict:
    """Blue Team: Optimize system (Prompt Evolution) to fix vulnerabilities."""
    from agents.core.blue_team_optimizer import BlueTeamOptimizer # Was BlueTeamOptimizer
    from agents.critics.fix_proposal import FixProposalAgent
    from agents.critics.verification import VerificationAgent, TestCase as VerificationTestCase
    from prompts.store import PromptGenomeStore
    from prompts.mutations import MutationEngine, create_mutation_from_fix

    if not state.get("enable_optimization", True):
        return {
            "current_phase": OrchestratorPhase.REPORTING, # Skip to next finding via a check? No, route logic handles it
            "messages": [{"role": "assistant", "content": "Optimization disabled."}]
        }

    findings = state["findings"]
    idx = state.get("current_finding_index", 0)
    current_finding = findings[idx]
    
    # Only optimize if severity is high enough (e.g. defense rate < 1.0)
    if current_finding.get("severity", 0) < 0.1:
        logger.info("Defense strong enough, skipping optimization.")
        return {
            "current_finding_index": idx + 1,
            "current_phase": OrchestratorPhase.ATTACK_PLANNING if idx + 1 < len(findings) else OrchestratorPhase.REPORTING,
             "messages": [{"role": "assistant", "content": "Defense sufficient, skipping patch."}]
        }

    # Initialize Blue Team Tools
    store = PromptGenomeStore(state.get("genome_store_path", "prompt_genomes.db"))
    optimizer = BlueTeamOptimizer(max_mutations_per_weakness=2) # Needs rename in file too
    mutation_engine = MutationEngine()
    verification_agent = VerificationAgent()
    
    genome = store.get_best() or store.ensure_default_exists()
    
    # generate proposals
    proposals = optimizer.map_weaknesses_batch([current_finding], max_total_mutations=3)
    
    applied_patches = []
    current_genome = genome
    
    # Convert attack cases to verification cases
    test_cases = []
    for ac in state.get("attack_cases", []):
         test_cases.append(VerificationTestCase(
             text=ac.get("text", ""),
             expected_entities=[{"entity_type": k, "text": v[0]} for k,v in ac.get("expected", {}).items() if v]
         ))
    
    # Apply patches loop
    for proposal in proposals:
        try:
             op, params = create_mutation_from_fix(proposal)
             mutated = mutation_engine.apply_mutation(current_genome, op, params)
             
             # Verify
             result = verification_agent.verify_fix(current_genome, mutated, test_cases)
             
             if result.improved and not result.regression_detected:
                 current_genome = mutated
                 applied_patches.append({
                     "patch": proposal.mutation.value,
                     "target": proposal.target,
                     "gain": result.delta_f1
                 })
        except Exception as e:
            logger.error(f"Patch failed: {e}")
            
    # Save if improved
    final_id = None
    if applied_patches:
        store.save(current_genome)
        final_id = current_genome.id
        
    next_idx = idx + 1
    next_phase = OrchestratorPhase.ATTACK_PLANNING if next_idx < len(findings) else OrchestratorPhase.REPORTING
    
    return {
        "applied_defenses": state.get("applied_defenses", []) + applied_patches,
        "final_genome_id": final_id,
        "current_finding_index": next_idx,
        "current_phase": next_phase,
        "messages": [{"role": "assistant", "content": f"Blue Team applied {len(applied_patches)} patches."}]
    }


def generate_report(state: RedBlueState) -> Dict:
    """Generate final Red-Blue engagement report."""
    findings = state["findings"]
    report = {
        "summary": {
            "vulnerabilities_found": len(findings),
            "defenses_applied": len(state.get("applied_defenses", [])),
            "final_genome_id": state.get("final_genome_id")
        },
        "details": findings,
        "defenses": state.get("applied_defenses", [])
    }
    
    return {
        "report": report,
        "current_phase": OrchestratorPhase.REPORTING,
        "messages": [{"role": "assistant", "content": "Engagement complete."}]
    }


# =============================================================================
# Graph Construction
# =============================================================================

def route_next_step(state: RedBlueState) -> Literal["attack_planning", "reporting"]:
    """Route based on whether more findings exist."""
    idx = state.get("current_finding_index", 0)
    findings = state.get("findings", [])
    if idx < len(findings):
        return "attack_planning"
    return "reporting"


def create_red_blue_graph() -> StateGraph:
    """Create the Red-Blue Orchestrator graph."""
    graph = StateGraph(RedBlueState)
    
    graph.add_node("baseline", run_baseline)
    graph.add_node("scan", scan_for_vulnerabilities)
    graph.add_node("red_team_plan", plan_attack)
    graph.add_node("red_team_execute", execute_red_team)
    graph.add_node("blue_team_analyze", analyze_root_cause)
    graph.add_node("blue_team_optimize", execute_blue_team)
    graph.add_node("report", generate_report)
    
    graph.set_entry_point("baseline")
    
    graph.add_edge("baseline", "scan")
    graph.add_edge("scan", "red_team_plan")
    graph.add_edge("red_team_plan", "red_team_execute")
    graph.add_edge("red_team_execute", "blue_team_analyze")
    graph.add_edge("blue_team_analyze", "blue_team_optimize")
    
    graph.add_conditional_edges(
        "blue_team_optimize",
        route_next_step,
        {
            "attack_planning": "red_team_plan",
            "reporting": "report"
        }
    )
    
    return graph


def run_adversarial_simulation(
    num_docs: int = 5,
    max_rounds: int = 3,
    confidence_threshold: float = 0.7,
    enable_evolution: bool = True,
) -> Dict:
    """Run the full Red-Blue adversarial simulation."""
    graph = create_red_blue_graph()
    app = graph.compile()
    
    initial_state = {
        "num_initial_docs": num_docs,
        "max_rounds": max_rounds,
        "confidence_threshold": confidence_threshold,
        "use_ministral": True,
        "ministral_model": DEFAULT_MODEL,
        "preset": "clinical",
        "current_phase": OrchestratorPhase.BASELINE,
        "current_round": 0,
        "findings": [],
        "applied_defenses": [],
        "enable_optimization": enable_evolution,
        "messages": [],
    }
    
    try:
        final_state = app.invoke(initial_state)
        return final_state.get("report", {})
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {"error": str(e)}
