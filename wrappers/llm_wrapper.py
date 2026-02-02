"""Prompt-Managed LLM Wrapper.

Wraps LLM-based recognizers with prompt genome management, enabling:
- Dynamic prompt configuration from PromptGenotype
- Automatic prompt building from structured genome components
- Seamless integration with OllamaNERecognizer's configurable prompts
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from prompts.genome import PromptGenotype
from prompts.store import PromptGenomeStore

if TYPE_CHECKING:
    from graph.state import DetectedEntity

logger = logging.getLogger(__name__)


class PromptManagedLLM:
    """Wrapper that manages prompt genomes for LLM recognizers.

    This class bridges the gap between PIIgent's prompt genome system
    and anoner's OllamaNERecognizer by:

    1. Loading/selecting prompt genomes from the store
    2. Building prompts from structured genome components
    3. Creating configured recognizers with the appropriate prompts

    Example:
        store = PromptGenomeStore("prompts.db")
        wrapper = PromptManagedLLM(genome_store=store)

        # Get recognizer with best-performing prompt
        recognizer = wrapper.create_recognizer()

        # Or with a specific genome
        recognizer = wrapper.create_recognizer(genome_id="genome_123")
    """

    def __init__(
        self,
        genome_store: PromptGenomeStore,
        active_genome_id: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        """Initialize the prompt-managed LLM wrapper.

        Args:
            genome_store: Store for prompt genomes
            active_genome_id: ID of genome to use (None = use best)
            ollama_url: URL of the Ollama server
            model: Ollama model name
            temperature: Model temperature
            timeout: Request timeout in seconds
        """
        self.genome_store = genome_store
        self.ollama_url = ollama_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        # Load active genome
        if active_genome_id:
            self.active_genome = genome_store.get(active_genome_id)
            if not self.active_genome:
                logger.warning(f"Genome {active_genome_id} not found, using best")
                self.active_genome = genome_store.get_best() or genome_store.ensure_default_exists()
        else:
            self.active_genome = genome_store.get_best() or genome_store.ensure_default_exists()

        logger.info(f"PromptManagedLLM initialized with genome {self.active_genome.id}")

    def set_active_genome(self, genome_id: str) -> bool:
        """Set the active genome by ID.

        Args:
            genome_id: ID of genome to activate

        Returns:
            True if genome was found and activated
        """
        genome = self.genome_store.get(genome_id)
        if genome:
            self.active_genome = genome
            logger.info(f"Activated genome {genome_id}")
            return True
        return False

    def create_recognizer(
        self,
        genome_id: Optional[str] = None,
        supported_entities: Optional[List[str]] = None,
        min_score: float = 0.4,
    ):
        """Create an OllamaNERecognizer with prompt from genome.

        Args:
            genome_id: Genome to use (None = use active genome)
            supported_entities: Entity types to extract
            min_score: Minimum confidence threshold

        Returns:
            Configured OllamaNERecognizer instance
        """
        # Import here to avoid circular dependency
        try:
            from presidio_analyzer.predefined_recognizers.ner import OllamaNERecognizer
        except ImportError:
            raise ImportError(
                "OllamaNERecognizer not available. "
                "Install from anoner/presidio-analyzer"
            )

        # Get genome to use
        if genome_id:
            genome = self.genome_store.get(genome_id)
            if not genome:
                logger.warning(f"Genome {genome_id} not found, using active")
                genome = self.active_genome
        else:
            genome = self.active_genome

        # Build prompts from genome
        system_prompt = genome.build_system_prompt()
        language_additions = {
            "de": genome.build_german_additions(),
        }

        # Create recognizer with configured prompts
        recognizer = OllamaNERecognizer(
            ollama_url=self.ollama_url,
            model=self.model,
            supported_entities=supported_entities,
            supported_language="de",
            temperature=self.temperature,
            timeout=self.timeout,
            min_score=min_score,
            system_prompt=system_prompt,
            language_additions=language_additions,
            name=f"PromptManaged_{genome.id[:20]}",
        )

        return recognizer

    def build_prompt(self, text: str) -> str:
        """Build full prompt from active genome (for debugging/display).

        Args:
            text: The text to analyze

        Returns:
            Full prompt string as it would be sent to the LLM
        """
        genome = self.active_genome

        parts = [
            genome.build_system_prompt(),
            genome.build_german_additions(),
            "",
            "Text to analyze:",
            text,
            "",
            "Return JSON array with all entities found:",
        ]

        return "\n".join(parts)

    def get_active_genome(self) -> PromptGenotype:
        """Get the currently active genome."""
        return self.active_genome

    def switch_to_best(self, metric: str = "exact_f1") -> Optional[PromptGenotype]:
        """Switch to the best-performing genome.

        Args:
            metric: Metric to rank by

        Returns:
            The new active genome, or None if no genomes with scores
        """
        best = self.genome_store.get_best(metric=metric)
        if best:
            self.active_genome = best
            logger.info(f"Switched to best genome {best.id} (metric={metric})")
        return best


class GenomeEvaluator:
    """Evaluates genomes and records fitness scores.

    Works with PromptManagedLLM to evaluate prompt performance
    and update fitness scores in the genome store.
    """

    def __init__(
        self,
        llm_wrapper: PromptManagedLLM,
        genome_store: PromptGenomeStore,
    ):
        """Initialize the genome evaluator.

        Args:
            llm_wrapper: The prompt-managed LLM wrapper
            genome_store: Store for prompt genomes
        """
        self.llm_wrapper = llm_wrapper
        self.genome_store = genome_store

    def evaluate_genome(
        self,
        genome_id: str,
        test_cases: List[Dict],
        test_set_id: str = "default",
    ) -> Dict[str, float]:
        """Evaluate a genome on test cases and record fitness.

        Args:
            genome_id: Genome to evaluate
            test_cases: List of test cases with 'text' and 'expected' keys
            test_set_id: Identifier for this test set

        Returns:
            Dict of metric name to score
        """
        # Import evaluation module
        from evaluation.metrics import MultiDimensionalMetrics

        recognizer = self.llm_wrapper.create_recognizer(genome_id=genome_id)

        # Run detection on all test cases
        all_expected = []
        all_detected = []

        for case in test_cases:
            text = case["text"]
            expected = case["expected"]  # List of expected entities

            # Run recognizer
            results = recognizer.analyze(text, language="de")
            detected = [
                {
                    "text": text[r.start:r.end],
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": r.score,
                }
                for r in results
            ]

            all_expected.append(expected)
            all_detected.append(detected)

        # Calculate metrics
        metrics = MultiDimensionalMetrics.calculate(
            expected_entities=all_expected,
            detected_entities=all_detected,
        )

        # Record fitness scores
        fitness_dict = metrics.to_fitness_dict()
        self.genome_store.record_fitness(
            genome_id=genome_id,
            metrics=fitness_dict,
            test_set_id=test_set_id,
        )

        return fitness_dict

    def compare_genomes(
        self,
        genome_ids: List[str],
        test_cases: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple genomes on the same test cases.

        Args:
            genome_ids: List of genome IDs to compare
            test_cases: Test cases to evaluate on

        Returns:
            Dict mapping genome ID to metrics dict
        """
        results = {}
        for genome_id in genome_ids:
            results[genome_id] = self.evaluate_genome(
                genome_id=genome_id,
                test_cases=test_cases,
            )
        return results
