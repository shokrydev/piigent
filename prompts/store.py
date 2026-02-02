"""Prompt Genome Store - SQLite-backed storage for prompt genomes.

Provides persistent storage for prompt genomes with support for:
- CRUD operations
- Lineage queries (get ancestors/descendants)
- Fitness-based retrieval (get best performers)
- Generation tracking
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from prompts.genome import PromptGenotype


class PromptGenomeStore:
    """SQLite-backed storage for prompt genomes.

    Stores prompt genomes persistently with support for lineage tracking,
    fitness queries, and generation-based retrieval.

    Example:
        store = PromptGenomeStore("prompts.db")
        genome = PromptGenotype.create_default()
        store.save(genome)

        # Get best performing genome
        best = store.get_best(metric="exact_f1")
    """

    def __init__(self, db_path: str = "prompt_genomes.db"):
        """Initialize the genome store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS genomes (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    generation INTEGER DEFAULT 0,
                    overall_fitness REAL DEFAULT 0.0,
                    eval_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    child_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    PRIMARY KEY (child_id, parent_id),
                    FOREIGN KEY (child_id) REFERENCES genomes(id),
                    FOREIGN KEY (parent_id) REFERENCES genomes(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS fitness_scores (
                    genome_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    score REAL NOT NULL,
                    test_set_id TEXT,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (genome_id, metric, test_set_id),
                    FOREIGN KEY (genome_id) REFERENCES genomes(id)
                )
            """)

            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_genomes_generation
                ON genomes(generation)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_genomes_fitness
                ON genomes(overall_fitness DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fitness_metric
                ON fitness_scores(metric, score DESC)
            """)

            conn.commit()

    def save(self, genome: PromptGenotype) -> None:
        """Save a genome to the store.

        Args:
            genome: The genome to save
        """
        now = datetime.now().isoformat()
        data = genome.to_json()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO genomes
                (id, data, generation, overall_fitness, eval_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                genome.id,
                data,
                genome.generation,
                genome.overall_fitness(),
                genome.eval_count,
                genome.created_at.isoformat(),
                now,
            ))

            # Save lineage
            for parent_id in genome.parent_ids:
                conn.execute("""
                    INSERT OR IGNORE INTO lineage (child_id, parent_id)
                    VALUES (?, ?)
                """, (genome.id, parent_id))

            # Save fitness scores
            for metric, score in genome.fitness_scores.items():
                conn.execute("""
                    INSERT OR REPLACE INTO fitness_scores
                    (genome_id, metric, score, test_set_id, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (genome.id, metric, score, "default", now))

            conn.commit()

    def get(self, genome_id: str) -> Optional[PromptGenotype]:
        """Get a genome by ID.

        Args:
            genome_id: The genome ID to retrieve

        Returns:
            The genome if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM genomes WHERE id = ?",
                (genome_id,)
            )
            row = cursor.fetchone()
            if row:
                return PromptGenotype.from_json(row[0])
        return None

    def get_best(
        self,
        metric: str = "exact_f1",
        min_evals: int = 1,
    ) -> Optional[PromptGenotype]:
        """Get the best-performing genome by a specific metric.

        Args:
            metric: The metric to rank by (e.g., "exact_f1", "recall")
            min_evals: Minimum number of evaluations required

        Returns:
            The highest-scoring genome, or None if none found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT g.data FROM genomes g
                INNER JOIN fitness_scores f ON g.id = f.genome_id
                WHERE f.metric = ? AND g.eval_count >= ?
                ORDER BY f.score DESC
                LIMIT 1
            """, (metric, min_evals))
            row = cursor.fetchone()
            if row:
                return PromptGenotype.from_json(row[0])

        # Fall back to any genome if no fitness scores exist
        return self.get_any()

    def get_any(self) -> Optional[PromptGenotype]:
        """Get any genome (useful for bootstrap)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM genomes LIMIT 1")
            row = cursor.fetchone()
            if row:
                return PromptGenotype.from_json(row[0])
        return None

    def get_top_k(
        self,
        metric: str = "exact_f1",
        k: int = 3,
        min_evals: int = 1,
    ) -> List[PromptGenotype]:
        """Get the top k genomes by a specific metric.

        Args:
            metric: The metric to rank by
            k: Number of genomes to return
            min_evals: Minimum number of evaluations required

        Returns:
            List of top-performing genomes
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT g.data FROM genomes g
                INNER JOIN fitness_scores f ON g.id = f.genome_id
                WHERE f.metric = ? AND g.eval_count >= ?
                ORDER BY f.score DESC
                LIMIT ?
            """, (metric, min_evals, k))
            return [PromptGenotype.from_json(row[0]) for row in cursor.fetchall()]

    def get_by_generation(self, generation: int) -> List[PromptGenotype]:
        """Get all genomes from a specific generation.

        Args:
            generation: The generation number

        Returns:
            List of genomes from that generation
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM genomes WHERE generation = ?",
                (generation,)
            )
            return [PromptGenotype.from_json(row[0]) for row in cursor.fetchall()]

    def get_ancestors(self, genome_id: str) -> List[PromptGenotype]:
        """Get all ancestors of a genome (lineage tree).

        Args:
            genome_id: The genome to get ancestors for

        Returns:
            List of all ancestor genomes
        """
        ancestors = []
        visited = set()
        to_visit = [genome_id]

        with sqlite3.connect(self.db_path) as conn:
            while to_visit:
                current_id = to_visit.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)

                cursor = conn.execute(
                    "SELECT parent_id FROM lineage WHERE child_id = ?",
                    (current_id,)
                )
                for row in cursor.fetchall():
                    parent_id = row[0]
                    to_visit.append(parent_id)
                    parent = self.get(parent_id)
                    if parent:
                        ancestors.append(parent)

        return ancestors

    def get_descendants(self, genome_id: str) -> List[PromptGenotype]:
        """Get all descendants of a genome.

        Args:
            genome_id: The genome to get descendants for

        Returns:
            List of all descendant genomes
        """
        descendants = []
        visited = set()
        to_visit = [genome_id]

        with sqlite3.connect(self.db_path) as conn:
            while to_visit:
                current_id = to_visit.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)

                cursor = conn.execute(
                    "SELECT child_id FROM lineage WHERE parent_id = ?",
                    (current_id,)
                )
                for row in cursor.fetchall():
                    child_id = row[0]
                    to_visit.append(child_id)
                    child = self.get(child_id)
                    if child:
                        descendants.append(child)

        return descendants

    def record_fitness(
        self,
        genome_id: str,
        metrics: Dict[str, float],
        test_set_id: str = "default",
    ) -> None:
        """Record fitness scores for a genome.

        Args:
            genome_id: The genome ID
            metrics: Dict of metric name to score
            test_set_id: Identifier for the test set used
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for metric, score in metrics.items():
                conn.execute("""
                    INSERT OR REPLACE INTO fitness_scores
                    (genome_id, metric, score, test_set_id, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (genome_id, metric, score, test_set_id, now))

            # Update genome's cached fitness scores
            genome = self.get(genome_id)
            if genome:
                genome.fitness_scores.update(metrics)
                genome.eval_count += 1
                self.save(genome)

            conn.commit()

    def get_lineage_graph(self, genome_id: str) -> Dict:
        """Get the lineage graph for visualization.

        Args:
            genome_id: The genome to build graph for

        Returns:
            Dict with nodes and edges for visualization
        """
        nodes = []
        edges = []
        visited = set()

        # Get ancestors and descendants
        genome = self.get(genome_id)
        if not genome:
            return {"nodes": [], "edges": []}

        all_related = [genome] + self.get_ancestors(genome_id) + self.get_descendants(genome_id)

        for g in all_related:
            if g.id not in visited:
                visited.add(g.id)
                nodes.append({
                    "id": g.id,
                    "generation": g.generation,
                    "fitness": g.overall_fitness(),
                    "eval_count": g.eval_count,
                })

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT child_id, parent_id FROM lineage")
            for row in cursor.fetchall():
                if row[0] in visited and row[1] in visited:
                    edges.append({
                        "from": row[1],
                        "to": row[0],
                    })

        return {"nodes": nodes, "edges": edges}

    def delete(self, genome_id: str) -> bool:
        """Delete a genome.

        Args:
            genome_id: The genome ID to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM genomes WHERE id = ?",
                (genome_id,)
            )
            conn.execute(
                "DELETE FROM lineage WHERE child_id = ? OR parent_id = ?",
                (genome_id, genome_id)
            )
            conn.execute(
                "DELETE FROM fitness_scores WHERE genome_id = ?",
                (genome_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Get total number of genomes in store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM genomes")
            return cursor.fetchone()[0]

    def list_all(self, limit: int = 100) -> List[PromptGenotype]:
        """List all genomes.

        Args:
            limit: Maximum number to return

        Returns:
            List of genomes
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM genomes ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            )
            return [PromptGenotype.from_json(row[0]) for row in cursor.fetchall()]

    def ensure_default_exists(self) -> PromptGenotype:
        """Ensure at least one genome exists, creating default if needed.

        Returns:
            The default genome (existing or newly created)
        """
        existing = self.get_any()
        if existing:
            return existing

        default = PromptGenotype.create_default()
        self.save(default)
        return default
