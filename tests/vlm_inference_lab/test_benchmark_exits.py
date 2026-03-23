import unittest
from unittest.mock import patch
from vlm_inference_lab.experiments.benchmark_serving import main as benchmark_main
import sys


class TestBenchmarkExits(unittest.TestCase):
    @patch("vlm_inference_lab.engines.vllm.VllmEngineAdapter.healthcheck")
    def test_benchmark_fails_on_unhealthy_backend(self, mock_health):
        mock_health.return_value = False

        # Test that benchmark exits with 1 if healthcheck fails
        with patch.object(sys, 'argv', ["benchmark_serving.py", "--backend", "vllm", "--url", "http://failed:8000/v1"]):
            with self.assertRaises(SystemExit) as cm:
                benchmark_main()
            self.assertEqual(cm.exception.code, 1)

    @patch("vlm_inference_lab.engines.vllm.VllmEngineAdapter.healthcheck")
    @patch("vlm_inference_lab.engines.vllm.VllmEngineAdapter.chat_completion")
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_benchmark_fails_on_request_error_with_flag(self, mock_open, mock_makedirs, mock_chat, mock_health):
        mock_health.return_value = True

        # Mock a failed completion result
        from vlm_inference_lab.engines.base import CompletionResult
        failed_result = CompletionResult(text="", prompt_tokens=0, completion_tokens=0, latency_ms=10,
                                         error="Test Error")
        mock_chat.return_value = failed_result

        # Test that benchmark exits with 1 if --fail-on-errors is set and requests fail
        test_args = ["benchmark_serving.py", "--backend", "vllm", "--num-requests", "1", "--fail-on-errors"]
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                benchmark_main()
            self.assertEqual(cm.exception.code, 1)

    @patch("vlm_inference_lab.engines.vllm.VllmEngineAdapter.healthcheck")
    @patch("vlm_inference_lab.engines.vllm.VllmEngineAdapter.chat_completion")
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_benchmark_succeeds_on_request_error_without_flag(self, mock_open, mock_makedirs, mock_chat, mock_health):
        mock_health.return_value = True

        # Mock a failed completion result
        from vlm_inference_lab.engines.base import CompletionResult
        failed_result = CompletionResult(text="", prompt_tokens=0, completion_tokens=0, latency_ms=10,
                                         error="Test Error")
        mock_chat.return_value = failed_result

        # Test that benchmark exits with 0 (default behavior of main if no exit is called)
        # However, our main() now has a default flow that continues.
        # If it doesn't call exit(1), it will finish normally (effectively exit 0)
        test_args = ["benchmark_serving.py", "--backend", "vllm", "--num-requests", "1"]
        with patch.object(sys, 'argv', test_args):
            # This should not raise SystemExit(1)
            try:
                benchmark_main()
            except SystemExit as e:
                if e.code != 0:
                    self.fail(f"benchmark_main() exited with {e.code} unexpectedly")


if __name__ == "__main__":
    unittest.main()
