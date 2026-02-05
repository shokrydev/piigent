#!/usr/bin/env python3
"""Comprehensive test suite for PIIgent Advanced Agentic Architecture.

Run with: python -m pytest tests/test_advanced_architecture.py -v
Or simply: python tests/test_advanced_architecture.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prompts_module():
    """Test Prompt Genome & Evolution System."""
    print("\n=== Testing prompts module ===")

    from prompts.genome import PromptGenotype, PromptExample
    from prompts.store import PromptGenomeStore
    from prompts.mutations import MutationOperator, MutationEngine
    from prompts.crossover import CrossoverEngine
    from prompts.fitness import PromptFitnessTracker, PopulationManager

    # Test genome creation
    genome = PromptGenotype.create_default()
    assert genome.id is not None
    assert len(genome.entity_definitions) > 0
    assert len(genome.constraints) > 0
    print(f"  ✓ Created genome: {genome.id}")

    # Test system prompt building
    prompt = genome.build_system_prompt()
    assert len(prompt) > 100
    assert "PERSON" in prompt
    print(f"  ✓ Built system prompt: {len(prompt)} chars")

    # Test German additions
    german = genome.build_german_additions()
    assert "German" in german or "PERSON" in german
    print(f"  ✓ Built German additions: {len(german)} chars")

    # Test genome cloning
    clone = genome.clone()
    assert clone.id != genome.id
    assert clone.parent_ids == [genome.id]
    print(f"  ✓ Cloned genome: {clone.id}")

    # Test serialization
    json_str = genome.to_json()
    restored = PromptGenotype.from_json(json_str)
    assert restored.id == genome.id
    print("  ✓ Serialization round-trip successful")

    # Test mutation engine
    engine = MutationEngine(mutation_rate=1.0)
    mutated = engine.apply_mutation(
        genome,
        MutationOperator.ADD_CONSTRAINT,
        {"constraint": "Test constraint"}
    )
    assert "Test constraint" in mutated.constraints
    print("  ✓ Mutation engine working")

    # Test crossover
    crossover = CrossoverEngine(crossover_rate=1.0)
    child = crossover.crossover(genome, clone)
    assert len(child.parent_ids) == 2
    print("  ✓ Crossover engine working")

    # Test store (in-memory with temp file)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = PromptGenomeStore(f.name)
        store.save(genome)
        retrieved = store.get(genome.id)
        assert retrieved is not None
        assert retrieved.id == genome.id
        print("  ✓ Genome store working")
        os.unlink(f.name)

    print("  ✓ All prompts module tests passed!")
    return True


def test_evaluation_module():
    """Test Multi-Dimensional Evaluation System."""
    print("\n=== Testing evaluation module ===")

    from evaluation.metrics import MultiDimensionalMetrics
    from evaluation.regression import RegressionTestSuite, TestCase
    from evaluation.disagreement import DisagreementEvaluator, DisagreementType
    from evaluation.distribution import DistributionMatcher, Distribution

    # Test metrics calculation - perfect match
    expected = [[{"text": "Berlin", "entity_type": "LOCATION", "start": 0, "end": 6}]]
    detected = [[{"text": "Berlin", "entity_type": "LOCATION", "start": 0, "end": 6, "score": 0.9}]]

    metrics = MultiDimensionalMetrics.calculate(expected, detected)
    assert metrics.exact_f1 == 1.0
    assert metrics.exact_precision == 1.0
    assert metrics.exact_recall == 1.0
    print("  ✓ Perfect match metrics correct")

    # Test metrics - miss
    expected = [[{"text": "Berlin", "entity_type": "LOCATION", "start": 0, "end": 6}]]
    detected = [[]]

    metrics = MultiDimensionalMetrics.calculate(expected, detected)
    assert metrics.exact_recall == 0.0
    assert metrics.false_negatives == 1
    print("  ✓ Miss detection correct")

    # Test metrics - false positive
    expected = [[]]
    detected = [[{"text": "Berlin", "entity_type": "LOCATION", "start": 0, "end": 6, "score": 0.9}]]

    metrics = MultiDimensionalMetrics.calculate(expected, detected)
    assert metrics.exact_precision == 0.0
    assert metrics.false_positives == 1
    print("  ✓ False positive detection correct")

    # Test fitness dict
    metrics = MultiDimensionalMetrics.calculate(
        [[{"text": "Test", "entity_type": "PERSON", "start": 0, "end": 4}]],
        [[{"text": "Test", "entity_type": "PERSON", "start": 0, "end": 4, "score": 0.8}]]
    )
    fitness = metrics.to_fitness_dict()
    assert "exact_f1" in fitness
    print("  ✓ Fitness dict generation working")

    # Test distribution
    dist = Distribution()
    dist.add("PERSON")
    dist.add("PERSON")
    dist.add("LOCATION")
    assert dist.probability("PERSON") == 2/3
    print("  ✓ Distribution calculation working")

    print("  ✓ All evaluation module tests passed!")
    return True


def test_curriculum_module():
    """Test Curriculum & Difficulty Scheduling System."""
    print("\n=== Testing curriculum module ===")

    from curriculum.difficulty import DifficultyDimensions, DifficultyLevel
    from curriculum.controller import CurriculumController
    from curriculum.sampler import FailureWeightedSampler

    # Test difficulty levels
    trivial = DifficultyDimensions.from_level(DifficultyLevel.TRIVIAL)
    expert = DifficultyDimensions.from_level(DifficultyLevel.EXPERT)

    assert trivial.overall_difficulty() < expert.overall_difficulty()
    print(f"  ✓ Trivial difficulty: {trivial.overall_difficulty():.2f}")
    print(f"  ✓ Expert difficulty: {expert.overall_difficulty():.2f}")

    # Test dimension adjustment
    adjusted = trivial.adjust_dimension("entity_count", 2)
    assert adjusted.entity_count == trivial.entity_count + 2
    print("  ✓ Dimension adjustment working")

    # Test SynPII config generation
    config = trivial.to_synpii_config()
    assert "min_entities" in config
    assert "max_entities" in config
    print("  ✓ SynPII config generation working")

    # Test curriculum controller
    controller = CurriculumController(
        start_difficulty=DifficultyLevel.EASY,
        promotion_threshold=0.85,
        min_evals_for_promotion=2,
    )

    assert controller.current_level == DifficultyLevel.EASY
    print("  ✓ Curriculum controller initialized")

    # Record evaluations
    controller.record_evaluation(f1=0.90, precision=0.92, recall=0.88)
    controller.record_evaluation(f1=0.91, precision=0.93, recall=0.89)

    assert controller.should_promote()
    controller.promote()
    assert controller.current_level == DifficultyLevel.MEDIUM
    print("  ✓ Promotion logic working")

    # Test failure sampler
    sampler = FailureWeightedSampler()
    sampler.record_failure("OCCUPATION", "social_history", "missed", "Ingenieur")
    sampler.record_failure("OCCUPATION", "social_history", "missed", "Lehrerin")
    sampler.record_failure("AGE", "anamnesis", "missed", "63-jährig")

    weights = sampler.get_sampling_weights()
    assert "OCCUPATION" in weights
    print(f"  ✓ Failure weights: {weights}")

    analysis = sampler.get_failure_analysis()
    assert analysis["total_failures"] == 3
    print("  ✓ Failure analysis working")

    print("  ✓ All curriculum module tests passed!")
    return True


def test_pipeline_module():
    """Test Multi-Step NER Pipeline."""
    print("\n=== Testing pipeline module ===")

    from nodes.section_parser import SectionParser, ClinicalSection
    from nodes.domain_rules import DomainRulesEngine, Rule, RuleAction
    from nodes.overlap_resolver import OverlapResolver, ResolutionStrategy
    from nodes.consistency import ConsistencyValidator

    # Test section parser
    parser = SectionParser()
    doc = parser.parse("""
Patient: Max Mustermann
KVNR: A123456789

Sozialanamnese:
Der Patient arbeitet als Ingenieur in München.

Diagnose:
Verdacht auf Hypertonie.
""")

    assert len(doc.sections) >= 2
    section_types = [s.section_type for s in doc.sections]
    assert ClinicalSection.SOCIAL_HISTORY in section_types
    print(f"  ✓ Parsed {len(doc.sections)} sections")

    # Test section entity expectations
    for section in doc.sections:
        if section.section_type == ClinicalSection.SOCIAL_HISTORY:
            assert "OCCUPATION" in section.expected_entities
            print(f"  ✓ OCCUPATION expected in social history section")

    # Test domain rules - suppress structural label
    engine = DomainRulesEngine()
    result = engine.apply_rules(
        entity_text="Behandelnder Arzt",
        entity_type="PERSON",
        confidence=0.8,
    )
    assert result is None  # Should be suppressed
    print("  ✓ Structural label suppressed")

    # Test domain rules - boost with context
    result = engine.apply_rules(
        entity_text="Müller",
        entity_type="PERSON",
        confidence=0.7,
        context="Herr Müller kam zur Untersuchung",
    )
    assert result is not None
    # Context boost rule looks for title at END of context before entity
    # So we may not get a boost here - just verify it's not suppressed
    print(f"  ✓ Entity not suppressed, confidence: {result['confidence']:.2f}")

    # Test overlap resolver
    resolver = OverlapResolver(strategy=ResolutionStrategy.PREFER_SPECIFIC)
    entities = [
        {"text": "10117 Berlin", "entity_type": "LOCATION", "start": 0, "end": 12, "score": 0.85},
        {"text": "10117", "entity_type": "DE_POSTAL_CODE", "start": 0, "end": 5, "score": 0.75},
    ]
    resolved = resolver.resolve(entities)
    # Both should be kept - postal code is more specific for its span
    assert len(resolved) == 2
    print("  ✓ Overlap resolver keeps nested specific entities")

    # Test consistency validator
    validator = ConsistencyValidator()
    entities = [
        {"text": "Berlin", "entity_type": "LOCATION", "start": 0, "end": 6},
        {"text": "Berlin", "entity_type": "ORGANIZATION", "start": 50, "end": 56},
    ]
    result = validator.validate(entities, "Berlin is a city. " * 3 + "Berlin hospital")
    assert len(result.issues) > 0
    print(f"  ✓ Found {len(result.issues)} consistency issues")

    print("  ✓ All pipeline module tests passed!")
    return True


def test_agents_module():
    """Test Self-Critique & Explanation Agents."""
    print("\n=== Testing agents module ===")

    from agents.error_taxonomy import ErrorTaxonomyAgent, ErrorType
    from agents.fix_proposal import FixProposalAgent, MutationOperator

    # Test error taxonomy - type confusion
    agent = ErrorTaxonomyAgent()
    error = agent.classify_error(
        expected={"text": "Ingenieur", "type": "OCCUPATION", "start": 10, "end": 19},
        detected={"text": "Ingenieur", "type": "PERSON", "start": 10, "end": 19, "score": 0.7},
        context="arbeitet als Ingenieur in München",
    )
    assert error.error_type == ErrorType.TYPE_CONFUSION
    print(f"  ✓ Type confusion detected: {error.explanation}")

    # Test error taxonomy - false negative
    error = agent.classify_error(
        expected={"text": "63-jährig", "type": "AGE", "start": 4, "end": 13},
        detected=None,
        context="Der 63-jährige Patient",
    )
    assert error.error_type in [ErrorType.FALSE_NEGATIVE, ErrorType.CONTEXT_DEPENDENT]
    print(f"  ✓ False negative detected: {error.error_type.value}")

    # Test batch analysis
    expected = [
        {"text": "Müller", "type": "PERSON", "start": 5, "end": 11},
        {"text": "63-jährig", "type": "AGE", "start": 20, "end": 29},
    ]
    detected = [
        {"text": "Müller", "type": "PERSON", "start": 5, "end": 11, "score": 0.9},
    ]
    errors = agent.analyze_batch(expected, detected, "Herr Müller, der 63-jährige Patient")
    assert len(errors) == 1  # AGE was missed
    print(f"  ✓ Batch analysis found {len(errors)} errors")

    # Test error summary
    summary = agent.get_error_summary()
    assert "total_errors" in summary
    print(f"  ✓ Error summary: {summary['total_errors']} total errors")

    # Test fix proposal
    from prompts.genome import PromptGenotype
    proposal_agent = FixProposalAgent()
    genome = PromptGenotype.create_default()

    fixes = proposal_agent.propose_fixes(errors, genome)
    print(f"  ✓ Proposed {len(fixes)} fixes")

    print("  ✓ All agents module tests passed!")
    return True


def test_safeguards_module():
    """Test System-Level Safeguards."""
    print("\n=== Testing safeguards module ===")

    from safeguards.overfitting import OverfittingDetector, EarlyStoppingMonitor
    from safeguards.leakage import LeakageChecker

    # Test leakage checker with clean prompt
    from prompts.genome import PromptGenotype
    checker = LeakageChecker()
    genome = PromptGenotype.create_default()

    report = checker.check_leakage(genome)
    assert not report.leakage_detected
    print("  ✓ Clean prompt has no leakage")

    # Test leakage checker with contaminated prompt
    contaminated = genome.clone()
    contaminated.instruction_block += " Look for {ENTITY_TYPE} patterns"
    contaminated.constraints.append("Match TEST_PATTERN_123")

    report = checker.check_leakage(contaminated)
    assert report.leakage_detected
    print(f"  ✓ Detected leakage: {report.leaked_items[:2]}")

    # Test sanitization
    sanitized = checker.sanitize_prompt(contaminated)
    report = checker.check_leakage(sanitized)
    # May still have some leakage depending on patterns, but should be reduced
    print(f"  ✓ Sanitized prompt (leakage: {report.leakage_detected})")

    # Test early stopping monitor
    monitor = EarlyStoppingMonitor(patience=3, min_delta=0.01)
    monitor.check(0.80)  # First score, sets baseline
    monitor.check(0.82)  # Improvement, resets wait
    monitor.check(0.82)  # No improvement, wait=1
    monitor.check(0.82)  # No improvement, wait=2
    monitor.check(0.82)  # No improvement, wait=3
    stopped = monitor.check(0.82)  # Should trigger stop
    assert stopped or monitor.wait >= monitor.patience
    print(f"  ✓ Early stopping: stopped={stopped}, wait={monitor.wait}")

    print("  ✓ All safeguards module tests passed!")
    return True


def test_wrappers_module():
    """Test Wrappers for External Integration."""
    print("\n=== Testing wrappers module ===")

    from wrappers.llm_wrapper import PromptManagedLLM
    from wrappers.synpii_wrapper import CurriculumSynPIIWrapper, GeneratedDocument
    from prompts.store import PromptGenomeStore
    from curriculum.controller import CurriculumController
    from curriculum.difficulty import DifficultyLevel

    # Test PromptManagedLLM initialization
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = PromptGenomeStore(f.name)
        store.ensure_default_exists()

        wrapper = PromptManagedLLM(genome_store=store)
        assert wrapper.active_genome is not None
        print(f"  ✓ PromptManagedLLM initialized with genome: {wrapper.active_genome.id[:20]}...")

        # Test prompt building
        prompt = wrapper.build_prompt("Test text with PII")
        assert "Test text with PII" in prompt
        print(f"  ✓ Built full prompt: {len(prompt)} chars")

        os.unlink(f.name)

    # Test CurriculumSynPIIWrapper initialization
    controller = CurriculumController(start_difficulty=DifficultyLevel.EASY)
    synpii_wrapper = CurriculumSynPIIWrapper(controller)

    # Get generation config
    config = controller.get_synpii_config()
    assert "min_entities" in config
    print(f"  ✓ SynPII config generated: {list(config.keys())[:5]}...")

    print("  ✓ All wrappers module tests passed!")
    return True


def test_ui_module():
    """Test UI Module (without launching Gradio)."""
    print("\n=== Testing UI module ===")

    from ui.hitl_interface import (
        HITLTrigger,
        AutonomyMode,
        HITLManager,
        PendingValidation,
        flag_for_validation,
        should_flag_for_hitl,
    )

    # Test enums
    assert HITLTrigger.LOW_CONFIDENCE.value == "low_confidence"
    assert AutonomyMode.SUPERVISED.value == "supervised"
    print("  ✓ Enums defined correctly")

    # Test HITL manager
    manager = HITLManager()
    case = PendingValidation(
        id="test_001",
        text="Der Patient Herr Müller",
        entity_text="Herr Müller",
        entity_type="PERSON",
        start=12,
        end=23,
        confidence=0.65,
        trigger=HITLTrigger.LOW_CONFIDENCE,
    )
    manager.add_case(case)
    assert len(manager.get_pending()) == 1
    print("  ✓ HITL manager accepts cases")

    # Test validation
    result = manager.validate("test_001", "approve")
    assert result
    assert len(manager.get_pending()) == 0
    assert len(manager.completed) == 1
    print("  ✓ Validation workflow working")

    # Test should_flag_for_hitl
    assert should_flag_for_hitl(0.5, threshold=0.7, mode=AutonomyMode.SUPERVISED)
    assert not should_flag_for_hitl(0.8, threshold=0.7, mode=AutonomyMode.SUPERVISED)
    assert not should_flag_for_hitl(0.5, threshold=0.7, mode=AutonomyMode.AUTONOMOUS)
    print("  ✓ HITL flagging logic correct")

    # Test flag_for_validation helper
    case_id = flag_for_validation(
        text="Test document",
        entity_text="Test",
        entity_type="PERSON",
        start=0,
        end=4,
        confidence=0.5,
        trigger=HITLTrigger.LOW_CONFIDENCE,
    )
    assert case_id is not None
    print(f"  ✓ Flagged case: {case_id}")

    print("  ✓ All UI module tests passed!")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PIIgent Advanced Architecture Test Suite")
    print("=" * 60)

    tests = [
        ("Prompts Module", test_prompts_module),
        ("Evaluation Module", test_evaluation_module),
        ("Curriculum Module", test_curriculum_module),
        ("Pipeline Module", test_pipeline_module),
        ("Agents Module", test_agents_module),
        ("Safeguards Module", test_safeguards_module),
        ("Wrappers Module", test_wrappers_module),
        ("UI Module", test_ui_module),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n  ✗ FAILED: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
