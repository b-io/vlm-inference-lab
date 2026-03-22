import time
import requests
from typing import Any, Dict, List, Optional
from .base import EngineAdapter, ChatMessage, CompletionResult


class VllmEngineAdapter(EngineAdapter):
    """An adapter for vLLM server via OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: Optional[str] = None):
        """Initializes the adapter with base URL and optional model name."""
        self.base_url = base_url.rstrip("/")
        self._model = model

    def healthcheck(self) -> bool:
        """Checks if the serving engine is healthy and reachable."""
        try:
            # Send a GET request to the models endpoint
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def chat_completion(self, messages: List[ChatMessage], **kwargs) -> CompletionResult:
        """Sends a chat completion request to the engine."""
        model = self.model_name()
        # Join messages into a single prompt for base models
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        # Heuristic: use /completions for base models, /chat/completions for chat/instruct models
        lower_model = model.lower()
        use_chat = any(token in lower_model for token in ["instruct", "chat", "it", "assistant"])

        # Record start time for latency calculation
        start_time = time.perf_counter()
        try:
            if use_chat:
                # Prepare payload for chat completions endpoint
                payload = {"model": model, "messages": [{"role": m.role, "content": m.content} for m in messages],
                        **kwargs, }
                response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=300, )
            else:
                # Prepare payload for standard completions endpoint
                payload = {"model": model, "prompt": prompt, **kwargs, }
                response = requests.post(f"{self.base_url}/completions", json=payload, timeout=300, )

            # Calculate latency in milliseconds
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                # Return result with error message on failure
                return CompletionResult(text="", prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
                        error=f"vLLM error: {response.text}", )

            # Parse JSON response and extract generated text
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            text = choice.get("message", {}).get("content")
            if text is None:
                text = choice.get("text", "")

            # Return the completion result
            return CompletionResult(text=text, prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0), latency_ms=latency_ms, )

        except requests.exceptions.RequestException as e:
            # Handle network errors and return results
            latency_ms = (time.perf_counter() - start_time) * 1000
            return CompletionResult(text="", prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
                    error=f"Network error: {str(e)}", )

    def metrics(self) -> Dict[str, Any]:
        """Fetches engine-specific metrics if available."""
        try:
            # Construct root URL for metrics endpoint
            root_url = self.base_url.replace("/v1", "")
            response = requests.get(f"{root_url}/metrics", timeout=5)
            if response.status_code == 200:
                # Return the first 500 characters of raw metrics
                return {"raw": response.text[:500]}
            return {}
        except Exception:
            return {}

    def model_name(self) -> str:
        """Returns the name of the model being served."""
        if self._model:
            return self._model

        try:
            # Query the models endpoint to get the first model ID
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    # Cache and return the model name
                    self._model = data["data"][0]["id"]
                    return self._model
        except Exception:
            pass
        return "unknown-model"
