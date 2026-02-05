import pytest
import numpy as np
from evaluation.metrics import MultiDimensionalMetrics
from evaluation.regression import RegressionTestSuite
from evaluation.disagreement import DisagreementEvaluator

class TestEvaluationModule:
    """Tests for the Multi-Dimensional Evaluation System."""

    def test_metrics_calculation(self):
        expected = [[{"text": "Berlin", "start": 0, "end": 6, "label": "LOCATION"}]]
        detected = [[{"text": "Berlin", "start": 0, "end": 6, "label": "LOCATION"}]]
        
        metrics = MultiDimensionalMetrics.calculate(expected, detected)
        assert metrics.exact_f1 == 1.0

    def test_regression_suite(self, tmp_path):
        db_path = tmp_path / "regression.db"
        suite = RegressionTestSuite(str(db_path))
        
        # Mock pipeline function
        def mock_pipeline(text):
            return {"entities": []}
            
        report = suite.run_regression_check(mock_pipeline)
        assert report.passed is not None

    def test_disagreement_evaluator(self):
        recognizer_a = [{"text": "Berlin", "start": 0, "end": 6, "entity_type": "LOCATION"}]
        recognizer_b = [{"text": "Berlin", "start": 0, "end": 6, "entity_type": "CITY"}] # Disagreement
        
        evaluator = DisagreementEvaluator(recognizers=[
            ("a", lambda x: recognizer_a), 
            ("b", lambda x: recognizer_b)
        ])
        cases = evaluator.find_disagreements("Berlin is nice.")
        assert len(cases) > 0
