import pytest
from unittest.mock import MagicMock, patch
import os

# Adjust the path to import LLMAgent correctly
# Assuming tests/test_agent.py is in the project root/tests
# and llm/langchain_agent.py is in the project root/llm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.langchain_agent import LLMAgent

@pytest.fixture
def mock_llm_agent():
    """Fixture to create an LLMAgent with a mocked LLM and chains."""
    with patch('llm.langchain_agent.OpenAI') as MockOpenAI:
        # Mock the OpenAI LLM instance
        mock_llm_instance = MagicMock()
        # Configure the mocked LLM to return specific responses for each chain
        mock_llm_instance.return_value.generate.return_value.generations = [[MagicMock(text="Mocked Diagnosis.")]]
        # Patch the overall_chain directly to control its output for the entire sequence
        with patch.object(LLMAgent, 'overall_chain') as mock_overall_chain:
            mock_overall_chain.return_value.invoke.return_value = {"report": "Mocked Diagnosis Report.", "diagnosis": "Mocked Diagnosis.", "recommendations": "Mocked Recommendations."} # Mocked output for the entire sequential chain

            agent = LLMAgent(openai_api_key="dummy_key")
            yield agent

def test_run_diagnosis_returns_non_empty_string(mock_llm_agent):
    """Test that run_diagnosis returns a non-empty string report."""
    anomaly_dict = {
        "anomaly_data": "High temperature on sensor 1, low pressure on sensor 2.",
        "context": "System operating under normal load.",
        "anomaly_details": "Sensor 1: 120C (threshold 100C), Sensor 2: 50psi (threshold 70psi)"
    }

    report = mock_llm_agent.run_diagnosis(anomaly_dict)

    assert isinstance(report, str)
    assert len(report) > 0
    assert "Mocked Diagnosis Report." in report 