"""Verification Agent.

Verifies whether proposed fixes actually improve performance
without causing regression.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from agents.critics.fix_proposal import ProposedFix
from evaluation.metrics import MultiDimensionalMetrics
from evaluation.regression import RegressionTestSuite
from prompts.genome import PromptGenotype


@dataclass
class VerificationResult:
    """Result of fix verification."""
    improved: bool
    delta_f1: float
    delta_precision: float
    delta_recall: float
    regression_detected: bool
    regression_details: List[Dict] = field(default_factory=list)
    recommendation: str = "reject"  # "accept", "reject", "review"
    original_metrics: Optional[MultiDimensionalMetrics] = None
    new_metrics: Optional[MultiDimensionalMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestCase:
    """A test case for verification."""
    text: str
    expected_entities: List[Dict]
    metadata: Dict = field(default_factory=dict)


class VerificationAgent:
    """Agent that verifies proposed fixes.

    Tests whether a proposed fix actually improves performance
    and checks for regression on the golden test set.

    Example:
        agent = VerificationAgent(regression_suite)

        result = agent.verify_fix(
            original_prompt=current_prompt,
            fixed_prompt=new_prompt,
            test_cases=test_cases,
        )

        if result.recommendation == "accept":
            print("Fix verified! Applying to production.")
    """

    def __init__(
        self,
        regression_suite: Optional[RegressionTestSuite] = None,
        min_improvement: float = 0.01,  # 1% minimum F1 improvement
        max_regression: float = 0.02,   # 2% maximum regression
    ):
        """Initialize the verification agent.

        Args:
            regression_suite: Suite for regression testing
            min_improvement: Minimum F1 improvement to accept
            max_regression: Maximum allowed regression
        """
        self.regression_suite = regression_suite
        self.min_improvement = min_improvement
        self.max_regression = max_regression

    def verify_fix(
        self,
        original_prompt: PromptGenotype,
        fixed_prompt: PromptGenotype,
        test_cases: List[TestCase],
        flow_factory: Optional[Callable[[PromptGenotype], Callable]] = None,
    ) -> VerificationResult:
        """Verify whether a fix improves performance.

        Args:
            original_prompt: The original prompt genome
            fixed_prompt: The fixed prompt genome
            test_cases: Test cases to evaluate on
            flow_factory: Function that creates flow from genome

        Returns:
            VerificationResult with recommendation
        """
        # Create flows for both prompts
        if flow_factory:
            original_flow = flow_factory(original_prompt)
            fixed_flow = flow_factory(fixed_prompt)
        else:
            # Use default evaluation
            original_flow = self._create_default_flow(original_prompt)
            fixed_flow = self._create_default_flow(fixed_prompt)

        # Run evaluation
        original_metrics = self._evaluate(original_flow, test_cases)
        new_metrics = self._evaluate(fixed_flow, test_cases)

        # Calculate deltas
        delta_f1 = new_metrics.exact_f1 - original_metrics.exact_f1
        delta_precision = new_metrics.exact_precision - original_metrics.exact_precision
        delta_recall = new_metrics.exact_recall - original_metrics.exact_recall

        # Check for improvement
        improved = delta_f1 >= self.min_improvement

        # Run regression check
        regression_detected = False
        regression_details = []

        if self.regression_suite and fixed_flow:
            regression_report = self.regression_suite.run_regression_check(fixed_flow)
            regression_detected = not regression_report.passed
            regression_details = regression_report.regressions

        # Determine recommendation
        if improved and not regression_detected:
            recommendation = "accept"
        elif improved and regression_detected:
            recommendation = "review"  # Improved but with regression
        elif not improved and delta_f1 >= -self.max_regression:
            recommendation = "review"  # No improvement but no regression
        else:
            recommendation = "reject"

        return VerificationResult(
            improved=improved,
            delta_f1=delta_f1,
            delta_precision=delta_precision,
            delta_recall=delta_recall,
            regression_detected=regression_detected,
            regression_details=regression_details,
            recommendation=recommendation,
            original_metrics=original_metrics,
            new_metrics=new_metrics,
        )

    def verify_multiple_fixes(
        self,
        original_prompt: PromptGenotype,
        fixes: List[ProposedFix],
        test_cases: List[TestCase],
        apply_fix_fn: Callable[[ProposedFix, PromptGenotype], PromptGenotype],
        flow_factory: Optional[Callable[[PromptGenotype], Callable]] = None,
    ) -> List[Tuple[ProposedFix, VerificationResult]]:
        """Verify multiple fixes and return results.

        Args:
            original_prompt: Original prompt genome
            fixes: List of proposed fixes
            test_cases: Test cases
            apply_fix_fn: Function to apply fix to genome
            flow_factory: Flow factory function

        Returns:
            List of (fix, verification_result) tuples
        """
        results = []

        for fix in fixes:
            fixed_prompt = apply_fix_fn(fix, original_prompt)
            result = self.verify_fix(
                original_prompt=original_prompt,
                fixed_prompt=fixed_prompt,
                test_cases=test_cases,
                flow_factory=flow_factory,
            )
            results.append((fix, result))

        return results

    def verify_cumulative_fixes(
        self,
        original_prompt: PromptGenotype,
        fixes: List[ProposedFix],
        test_cases: List[TestCase],
        apply_fix_fn: Callable[[ProposedFix, PromptGenotype], PromptGenotype],
        flow_factory: Optional[Callable[[PromptGenotype], Callable]] = None,
    ) -> Tuple[PromptGenotype, List[Tuple[ProposedFix, VerificationResult]]]:
        """Apply and verify fixes cumulatively, keeping only improvements.

        Args:
            original_prompt: Original prompt genome
            fixes: List of proposed fixes (should be priority-sorted)
            test_cases: Test cases
            apply_fix_fn: Function to apply fix
            flow_factory: Flow factory

        Returns:
            Tuple of (final_prompt, list of applied (fix, result))
        """
        current_prompt = original_prompt
        applied_fixes = []

        for fix in fixes:
            fixed_prompt = apply_fix_fn(fix, current_prompt)
            result = self.verify_fix(
                original_prompt=current_prompt,
                fixed_prompt=fixed_prompt,
                test_cases=test_cases,
                flow_factory=flow_factory,
            )

            if result.recommendation == "accept":
                current_prompt = fixed_prompt
                applied_fixes.append((fix, result))

        return current_prompt, applied_fixes

    def _evaluate(
        self,
        flow: Callable,
        test_cases: List[TestCase],
    ) -> MultiDimensionalMetrics:
        """Evaluate flow on test cases.

        Args:
            flow: Detection flow function
            test_cases: Test cases

        Returns:
            MultiDimensionalMetrics
        """
        expected_all = []
        detected_all = []

        for case in test_cases:
            try:
                detected = flow(case.text)
                if isinstance(detected, list):
                    detected_entities = detected
                else:
                    # Handle different return formats
                    detected_entities = detected.get("entities", []) if isinstance(detected, dict) else []
            except Exception:
                detected_entities = []

            expected_all.append(case.expected_entities)
            detected_all.append(detected_entities)

        return MultiDimensionalMetrics.calculate(
            expected_entities=expected_all,
            detected_entities=detected_all,
        )

    def _create_default_flow(
        self,
        prompt: PromptGenotype,
    ) -> Callable:
        """Create a simple evaluation flow.

        This is a placeholder - in real use, flow_factory should be provided.
        """
        def mock_flow(text: str) -> List[Dict]:
            # Return empty for mock - real implementation uses PromptManagedLLM
            return []

        return mock_flow

    def a_b_test(
        self,
        prompt_a: PromptGenotype,
        prompt_b: PromptGenotype,
        test_cases: List[TestCase],
        flow_factory: Callable[[PromptGenotype], Callable],
        num_trials: int = 3,
    ) -> Dict:
        """Run A/B test between two prompts.

        Args:
            prompt_a: First prompt (typically original)
            prompt_b: Second prompt (typically modified)
            test_cases: Test cases
            flow_factory: Flow factory
            num_trials: Number of evaluation trials

        Returns:
            A/B test results with statistical significance
        """
        a_scores = []
        b_scores = []

        for _ in range(num_trials):
            flow_a = flow_factory(prompt_a)
            flow_b = flow_factory(prompt_b)

            metrics_a = self._evaluate(flow_a, test_cases)
            metrics_b = self._evaluate(flow_b, test_cases)

            a_scores.append(metrics_a.exact_f1)
            b_scores.append(metrics_b.exact_f1)

        # Calculate statistics
        a_mean = sum(a_scores) / len(a_scores)
        b_mean = sum(b_scores) / len(b_scores)

        a_var = sum((s - a_mean) ** 2 for s in a_scores) / len(a_scores)
        b_var = sum((s - b_mean) ** 2 for s in b_scores) / len(b_scores)

        # Simple significance test (would use proper stats in production)
        effect_size = (b_mean - a_mean) / ((a_var + b_var) ** 0.5 + 1e-8)
        significant = abs(effect_size) > 0.5  # Cohen's d threshold

        return {
            "prompt_a_mean_f1": a_mean,
            "prompt_b_mean_f1": b_mean,
            "delta": b_mean - a_mean,
            "effect_size": effect_size,
            "significant": significant,
            "winner": "B" if b_mean > a_mean and significant else ("A" if a_mean > b_mean and significant else "tie"),
            "trials": num_trials,
        }
