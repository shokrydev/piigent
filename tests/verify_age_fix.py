
import sys
import unittest
from unittest.mock import MagicMock, patch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add repo root to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.core.detection_coordinator import _run_ministral

class TestAgeDetectionFix(unittest.TestCase):
    @patch('presidio_analyzer.predefined_recognizers.OllamaNERecognizer')
    def test_run_ministral_includes_age(self, MockOllamaRecognizer):
        # Setup mock
        mock_instance = MockOllamaRecognizer.return_value
        mock_instance.analyze.return_value = [] # Return empty list, we just want to check the call args
        
        # Test input
        text = "Patient ist 45 Jahre alt."
        model = "ministral-test"
        url = "http://localhost:11434"
        
        # Run function
        _run_ministral(text, model, url)
        
        # Verify analyze was called with AGE in entities
        call_args = mock_instance.analyze.call_args
        self.assertIsNotNone(call_args, "analyze() should have been called")
        
        _, kwargs = call_args
        entities = kwargs.get('entities')
        
        logger.info(f"Entities passed to analyze: {entities}")
        
        self.assertIn("AGE", entities, "AGE should be in the list of entities passed to recognizer.analyze")
        
        print("SUCCESS: 'AGE' was found in the entities list!")

if __name__ == '__main__':
    unittest.main()
