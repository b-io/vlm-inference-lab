from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: str
    content: str

@dataclass
class CompletionResult:
    """A container for completion output and associated metadata."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: Optional[str] = None

class EngineAdapter(ABC):
    """An abstract base class for inference engine adapters."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Checks if the serving engine is healthy and reachable."""
        pass

    @abstractmethod
    def chat_completion(
        self, 
        messages: List[ChatMessage], 
        **kwargs
    ) -> CompletionResult:
        """Sends a chat completion request to the engine."""
        pass

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """Fetches engine-specific metrics if available."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Returns the name of the model being served."""
        pass
