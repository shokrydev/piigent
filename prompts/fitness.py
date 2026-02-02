"""Prompt Fitness Tracking.

Tracks prompt performance over time and provides
utilities for selecting high-performing prompts.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompts.genome import PromptGenotype


@dataclass
class FitnessRecord:
    """A single fitness evaluation record."""
    genome_id: str
    metric: str
    score: float
    test_set_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class PromptFitnessTracker:
    """Tracks prompt performance over time.

    Stores fitness scores from evaluations and provides utilities
    for analyzing performance trends and selecting high performers.

    Example:
        tracker = PromptFitnessTracker()

        # Record evaluation
        tracker.record_evaluation(
            prompt_id="genome_123",
            metrics={"exact_f1": 0.82, "recall": 0.85},
            test_set_id="clinical_v1",
        )

        # Get high performers
        top_prompts = tracker.identify_high_performers(
            metric="exact_f1",
            top_k=3,
        )
    """

    def __init__(self, db_path: str = "prompt_fitness.db"):
        """Initialize the fitness tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fitness_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genome_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    score REAL NOT NULL,
                    test_set_id TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fitness_genome
                ON fitness_records(genome_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fitness_metric
                ON fitness_records(metric, score DESC)
            """)

            conn.commit()

    def record_evaluation(
        self,
        prompt_id: str,
        metrics: Dict[str, float],
        test_set_id: str = "default",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record evaluation results for a prompt.

        Args:
            prompt_id: The prompt genome ID
            metrics: Dict of metric name to score
            test_set_id: Identifier for the test set used
            metadata: Additional metadata
        """
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            for metric, score in metrics.items():
                conn.execute("""
                    INSERT INTO fitness_records
                    (genome_id, metric, score, test_set_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (prompt_id, metric, score, test_set_id, now, metadata_json))
            conn.commit()

    def get_fitness_history(
        self,
        prompt_id: str,
        metric: Optional[str] = None,
    ) -> List[FitnessRecord]:
        """Get fitness history for a prompt.

        Args:
            prompt_id: The prompt genome ID
            metric: Optional specific metric to filter

        Returns:
            List of fitness records
        """
        with sqlite3.connect(self.db_path) as conn:
            if metric:
                cursor = conn.execute("""
                    SELECT genome_id, metric, score, test_set_id, timestamp, metadata
                    FROM fitness_records
                    WHERE genome_id = ? AND metric = ?
                    ORDER BY timestamp
                """, (prompt_id, metric))
            else:
                cursor = conn.execute("""
                    SELECT genome_id, metric, score, test_set_id, timestamp, metadata
                    FROM fitness_records
                    WHERE genome_id = ?
                    ORDER BY timestamp
                """, (prompt_id,))

            return [
                FitnessRecord(
                    genome_id=row[0],
                    metric=row[1],
                    score=row[2],
                    test_set_id=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                for row in cursor.fetchall()
            ]

    def get_latest_fitness(
        self,
        prompt_id: str,
    ) -> Dict[str, float]:
        """Get the latest fitness scores for a prompt.

        Args:
            prompt_id: The prompt genome ID

        Returns:
            Dict of metric to latest score
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric, score FROM fitness_records
                WHERE genome_id = ?
                AND timestamp = (
                    SELECT MAX(timestamp) FROM fitness_records
                    WHERE genome_id = ? AND metric = fitness_records.metric
                )
            """, (prompt_id, prompt_id))

            return {row[0]: row[1] for row in cursor.fetchall()}

    def identify_high_performers(
        self,
        metric: str = "exact_f1",
        min_evals: int = 1,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Identify top-performing prompts.

        Args:
            metric: Metric to rank by
            min_evals: Minimum evaluations required
            top_k: Number of top prompts to return

        Returns:
            List of (prompt_id, score) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT genome_id, MAX(score) as best_score, COUNT(*) as eval_count
                FROM fitness_records
                WHERE metric = ?
                GROUP BY genome_id
                HAVING eval_count >= ?
                ORDER BY best_score DESC
                LIMIT ?
            """, (metric, min_evals, top_k))

            return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_improvement_rate(
        self,
        prompt_id: str,
        metric: str = "exact_f1",
    ) -> float:
        """Calculate improvement rate over evaluations.

        Args:
            prompt_id: The prompt genome ID
            metric: Metric to analyze

        Returns:
            Improvement rate (positive = improving)
        """
        history = self.get_fitness_history(prompt_id, metric)

        if len(history) < 2:
            return 0.0

        scores = [r.score for r in history]
        n = len(scores)

        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_lineage_performance(
        self,
        prompt_ids: List[str],
        metric: str = "exact_f1",
    ) -> Dict[str, Dict]:
        """Get performance comparison across a lineage.

        Args:
            prompt_ids: List of prompt IDs in the lineage
            metric: Metric to compare

        Returns:
            Dict mapping prompt_id to performance stats
        """
        results = {}

        for prompt_id in prompt_ids:
            history = self.get_fitness_history(prompt_id, metric)
            if history:
                scores = [r.score for r in history]
                results[prompt_id] = {
                    "best": max(scores),
                    "latest": scores[-1],
                    "mean": sum(scores) / len(scores),
                    "eval_count": len(scores),
                    "improvement_rate": self.get_improvement_rate(prompt_id, metric),
                }

        return results

    def get_global_statistics(self) -> Dict:
        """Get global statistics across all prompts.

        Returns:
            Statistics dict
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total evaluations
            cursor = conn.execute("SELECT COUNT(*) FROM fitness_records")
            total_evals = cursor.fetchone()[0]

            # Unique prompts
            cursor = conn.execute("SELECT COUNT(DISTINCT genome_id) FROM fitness_records")
            unique_prompts = cursor.fetchone()[0]

            # Best scores per metric
            cursor = conn.execute("""
                SELECT metric, MAX(score), genome_id
                FROM fitness_records
                GROUP BY metric
            """)
            best_scores = {row[0]: {"score": row[1], "genome_id": row[2]} for row in cursor.fetchall()}

            return {
                "total_evaluations": total_evals,
                "unique_prompts": unique_prompts,
                "best_scores": best_scores,
            }

    def prune_old_records(
        self,
        keep_days: int = 30,
        keep_top_n: int = 10,
    ) -> int:
        """Prune old fitness records.

        Args:
            keep_days: Keep records from the last N days
            keep_top_n: Always keep top N performers

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now()
        # Implementation would calculate cutoff and delete old records
        # while preserving top performers
        return 0  # Placeholder


class PopulationManager:
    """Manages a population of prompt genomes for evolution.

    Handles selection, replacement, and diversity maintenance.
    """

    def __init__(
        self,
        fitness_tracker: PromptFitnessTracker,
        population_size: int = 20,
        elite_count: int = 3,  # Always keep top N
    ):
        """Initialize the population manager.

        Args:
            fitness_tracker: Fitness tracker instance
            population_size: Target population size
            elite_count: Number of elites to always preserve
        """
        self.fitness_tracker = fitness_tracker
        self.population_size = population_size
        self.elite_count = elite_count
        self.population: List[str] = []  # List of genome IDs

    def add_to_population(self, genome_id: str) -> None:
        """Add a genome to the population."""
        if genome_id not in self.population:
            self.population.append(genome_id)

    def select_parents(
        self,
        count: int = 2,
        metric: str = "exact_f1",
        method: str = "tournament",
    ) -> List[str]:
        """Select parents for breeding.

        Args:
            count: Number of parents to select
            metric: Fitness metric for selection
            method: Selection method ("tournament", "roulette", "rank")

        Returns:
            List of selected genome IDs
        """
        if method == "tournament":
            return self._tournament_selection(count, metric)
        elif method == "roulette":
            return self._roulette_selection(count, metric)
        else:
            return self._rank_selection(count, metric)

    def _tournament_selection(
        self,
        count: int,
        metric: str,
        tournament_size: int = 3,
    ) -> List[str]:
        """Tournament selection."""
        import random

        selected = []

        for _ in range(count):
            # Random tournament
            candidates = random.sample(
                self.population,
                min(tournament_size, len(self.population))
            )

            # Select best in tournament
            best = None
            best_score = -1

            for genome_id in candidates:
                fitness = self.fitness_tracker.get_latest_fitness(genome_id)
                score = fitness.get(metric, 0)
                if score > best_score:
                    best_score = score
                    best = genome_id

            if best:
                selected.append(best)

        return selected

    def _roulette_selection(self, count: int, metric: str) -> List[str]:
        """Roulette wheel selection."""
        import random

        # Get fitness scores
        fitnesses = []
        for genome_id in self.population:
            fitness = self.fitness_tracker.get_latest_fitness(genome_id)
            fitnesses.append((genome_id, fitness.get(metric, 0.1)))

        total_fitness = sum(f[1] for f in fitnesses)

        selected = []
        for _ in range(count):
            pick = random.uniform(0, total_fitness)
            current = 0
            for genome_id, score in fitnesses:
                current += score
                if current >= pick:
                    selected.append(genome_id)
                    break

        return selected

    def _rank_selection(self, count: int, metric: str) -> List[str]:
        """Rank-based selection."""
        import random

        # Get fitness scores and rank
        fitnesses = []
        for genome_id in self.population:
            fitness = self.fitness_tracker.get_latest_fitness(genome_id)
            fitnesses.append((genome_id, fitness.get(metric, 0)))

        fitnesses.sort(key=lambda x: -x[1])

        # Assign rank-based probabilities
        n = len(fitnesses)
        weights = [n - i for i in range(n)]  # Higher rank = higher weight
        total = sum(weights)
        probs = [w / total for w in weights]

        selected = random.choices(
            [f[0] for f in fitnesses],
            weights=probs,
            k=count,
        )

        return selected

    def replace_weakest(
        self,
        new_genome_ids: List[str],
        metric: str = "exact_f1",
    ) -> List[str]:
        """Replace weakest members with new genomes.

        Args:
            new_genome_ids: New genomes to add
            metric: Fitness metric

        Returns:
            List of removed genome IDs
        """
        removed = []

        # Get current fitness rankings
        fitnesses = []
        for genome_id in self.population:
            fitness = self.fitness_tracker.get_latest_fitness(genome_id)
            fitnesses.append((genome_id, fitness.get(metric, 0)))

        fitnesses.sort(key=lambda x: -x[1])

        # Identify elites (always keep)
        elites = {f[0] for f in fitnesses[:self.elite_count]}

        # Remove weakest non-elites
        for new_id in new_genome_ids:
            if len(self.population) >= self.population_size:
                # Find weakest non-elite
                for genome_id, _ in reversed(fitnesses):
                    if genome_id not in elites and genome_id in self.population:
                        self.population.remove(genome_id)
                        removed.append(genome_id)
                        break

            self.population.append(new_id)

        return removed
