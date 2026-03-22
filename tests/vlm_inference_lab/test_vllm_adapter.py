import unittest
from unittest.mock import patch, MagicMock
from vlm_inference_lab.engines.vllm import VllmEngineAdapter
from vlm_inference_lab.engines.base import ChatMessage, CompletionResult

class TestVllmEngineAdapterHeuristic(unittest.TestCase):
    def setUp(self):
        self.adapter = VllmEngineAdapter(base_url="http://test:8000/v1")

    @patch("requests.post")
    def test_chat_model_heuristic(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "chat response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_post.return_value = mock_response

        # Test with an "instruct" model
        self.adapter._model = "llama-3-8b-instruct"
        messages = [ChatMessage(role="user", content="hello")]
        
        result = self.adapter.chat_completion(messages)
        
        # Verify it used /chat/completions
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("/chat/completions", args[0])
        self.assertEqual(result.text, "chat response")

    @patch("requests.post")
    def test_base_model_heuristic(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "base response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_post.return_value = mock_response

        # Test with a base model
        self.adapter._model = "llama-3-8b"
        messages = [ChatMessage(role="user", content="hello")]
        
        result = self.adapter.chat_completion(messages)
        
        # Verify it used /completions
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("/completions", args[0])
        self.assertNotIn("/chat/", args[0])
        self.assertEqual(result.text, "base response")
        
        # Verify prompt format for base model
        payload = kwargs["json"]
        self.assertEqual(payload["prompt"], "user: hello")

    @patch("requests.get")
    def test_healthcheck(self, mock_get):
        mock_get.return_value.status_code = 200
        self.assertTrue(self.adapter.healthcheck())
        mock_get.assert_called_with("http://test:8000/v1/models", timeout=5)

if __name__ == "__main__":
    unittest.main()
