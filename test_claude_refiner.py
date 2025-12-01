import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flashrag.refiner.claude_refiner import ClaudeRefiner

class MockItem:
    def __init__(self, question, retrieval_result):
        self.question = question
        self.retrieval_result = retrieval_result

def test_claude_refiner():
    print("Testing ClaudeRefiner...")
    
    # Mock config
    config = {
        "refiner_name": "claude",
        "refiner_model_path": "claude-sonnet-4-20250514-v1:0",
        "device": "cpu",
        "refiner_input_prompt_flag": False
    }

    # Mock environment variable
    with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "fake_key"}):
        # Mock OpenAI client
        with patch('flashrag.refiner.claude_refiner.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            # Mock response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Refined content"
            mock_client.chat.completions.create.return_value = mock_response

            # Instantiate refiner
            refiner = ClaudeRefiner(config)
            
            # Create dummy data
            item = MockItem(
                question="What is the capital of France?",
                retrieval_result=[
                    {"contents": "Title: Paris\nParis is the capital of France."},
                    {"contents": "Title: London\nLondon is the capital of UK."}
                ]
            )
            dataset = [item]

            # Run batch_run
            output = refiner.batch_run(dataset)

            # Verify output
            assert len(output) == 1
            assert output[0] == "Refined content"
            
            # Verify API call
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs['model'] == "claude-sonnet-4-20250514-v1:0"
            assert len(call_args.kwargs['messages']) == 2
            assert "Paris is the capital of France" in call_args.kwargs['messages'][1]['content']

            print("ClaudeRefiner test passed!")

if __name__ == "__main__":
    test_claude_refiner()
