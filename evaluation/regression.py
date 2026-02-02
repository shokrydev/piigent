"""Regression Test Suite.

Frozen test set to prevent regression during prompt evolution.
Tracks baseline scores and alerts when performance drops.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from evaluation.metrics import MultiDimensionalMetrics


@dataclass
class TestCase:
    """A single test case with text and expected entities."""
    id: str
    text: str
    expected_entities: List[Dict]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "expected_entities": self.expected_entities,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TestCase":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            expected_entities=data["expected_entities"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegressionReport:
    """Report from regression testing."""
    passed: bool
    regressions: List[Dict]  # List of {metric, baseline, current, delta}
    improvements: List[Dict]  # List of {metric, baseline, current, delta}
    baseline_scores: Dict[str, float]
    current_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Regression Test Report ===",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
        ]

        if self.regressions:
            lines.append("REGRESSIONS (performance decreased):")
            for reg in self.regressions:
                lines.append(
                    f"  {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f} "
                    f"({reg['delta']:+.3f})"
                )

        if self.improvements:
            lines.append("\nIMPROVEMENTS (performance increased):")
            for imp in self.improvements:
                lines.append(
                    f"  {imp['metric']}: {imp['baseline']:.3f} → {imp['current']:.3f} "
                    f"({imp['delta']:+.3f})"
                )

        return "\n".join(lines)


class RegressionTestSuite:
    """Frozen test set to prevent regression.

    Maintains a golden set of test cases that should never regress.
    Compares current performance against established baselines.

    Example:
        suite = RegressionTestSuite("regression.db")

        # Add test cases (usually done once)
        suite.add_test_case(TestCase(
            id="clinical_1",
            text="Herr Bauer, KVNR: A123456789",
            expected_entities=[
                {"text": "Herr Bauer", "entity_type": "PERSON", "start": 0, "end": 10},
                {"text": "A123456789", "entity_type": "DE_KVNR", "start": 18, "end": 28},
            ],
        ))

        # Run regression check
        report = suite.run_regression_check(pipeline_fn)
        if not report.passed:
            print("REGRESSION DETECTED!")
    """

    def __init__(
        self,
        db_path: str = "regression_tests.db",
        tolerance: float = 0.02,  # 2% tolerance
    ):
        """Initialize the regression test suite.

        Args:
            db_path: Path to SQLite database
            tolerance: Acceptable performance drop (e.g., 0.02 = 2%)
        """
        self.db_path = Path(db_path)
        self.tolerance = tolerance
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_cases (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    metric TEXT PRIMARY KEY,
                    score REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metrics TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.commit()

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the golden set.

        Args:
            test_case: The test case to add
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO test_cases (id, data, created_at)
                VALUES (?, ?, ?)
            """, (
                test_case.id,
                json.dumps(test_case.to_dict()),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_test_cases(self) -> List[TestCase]:
        """Get all test cases in the golden set."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM test_cases")
            return [TestCase.from_dict(json.loads(row[0])) for row in cursor.fetchall()]

    def remove_test_case(self, test_id: str) -> bool:
        """Remove a test case from the golden set.

        Args:
            test_id: ID of test case to remove

        Returns:
            True if removed, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM test_cases WHERE id = ?",
                (test_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_baselines(self) -> Dict[str, float]:
        """Get current baseline scores."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT metric, score FROM baselines")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def update_baseline(
        self,
        pipeline_fn: Callable[[str], List[Dict]],
    ) -> Dict[str, float]:
        """Update baseline scores from current pipeline performance.

        Args:
            pipeline_fn: Function that takes text and returns detected entities

        Returns:
            New baseline scores
        """
        test_cases = self.get_test_cases()
        if not test_cases:
            return {}

        # Run evaluation
        expected = [[tc.expected_entities] for tc in test_cases]
        detected = []

        for tc in test_cases:
            results = pipeline_fn(tc.text)
            detected.append([results])

        # Calculate metrics
        metrics = MultiDimensionalMetrics.calculate(
            expected_entities=[tc.expected_entities for tc in test_cases],
            detected_entities=detected,
        )

        # Update baselines
        new_baselines = metrics.to_fitness_dict()
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for metric, score in new_baselines.items():
                conn.execute("""
                    INSERT OR REPLACE INTO baselines (metric, score, updated_at)
                    VALUES (?, ?, ?)
                """, (metric, score, now))
            conn.commit()

        return new_baselines

    def run_regression_check(
        self,
        pipeline_fn: Callable[[str], List[Dict]],
    ) -> RegressionReport:
        """Run regression check against baseline.

        Args:
            pipeline_fn: Function that takes text and returns detected entities

        Returns:
            RegressionReport with pass/fail and details
        """
        test_cases = self.get_test_cases()
        baselines = self.get_baselines()

        if not test_cases:
            return RegressionReport(
                passed=True,
                regressions=[],
                improvements=[],
                baseline_scores={},
                current_scores={},
            )

        # Run evaluation
        detected = []
        for tc in test_cases:
            results = pipeline_fn(tc.text)
            detected.append(results)

        metrics = MultiDimensionalMetrics.calculate(
            expected_entities=[tc.expected_entities for tc in test_cases],
            detected_entities=detected,
        )

        current_scores = metrics.to_fitness_dict()

        # Compare against baselines
        regressions = []
        improvements = []

        for metric, current in current_scores.items():
            baseline = baselines.get(metric)
            if baseline is None:
                continue

            delta = current - baseline

            if delta < -self.tolerance:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline,
                    "current": current,
                    "delta": delta,
                })
            elif delta > self.tolerance:
                improvements.append({
                    "metric": metric,
                    "baseline": baseline,
                    "current": current,
                    "delta": delta,
                })

        passed = len(regressions) == 0

        # Record in history
        report = RegressionReport(
            passed=passed,
            regressions=regressions,
            improvements=improvements,
            baseline_scores=baselines,
            current_scores=current_scores,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluation_history (metrics, passed, timestamp)
                VALUES (?, ?, ?)
            """, (
                json.dumps(current_scores),
                1 if passed else 0,
                datetime.now().isoformat(),
            ))
            conn.commit()

        return report

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluation history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of historical evaluation results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics, passed, timestamp
                FROM evaluation_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [
                {
                    "metrics": json.loads(row[0]),
                    "passed": bool(row[1]),
                    "timestamp": row[2],
                }
                for row in cursor.fetchall()
            ]

    def count_test_cases(self) -> int:
        """Get number of test cases in golden set."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test_cases")
            return cursor.fetchone()[0]
