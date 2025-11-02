"""
Base class for all model wrappers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Container for model generation response"""
    text: str
    latency: float
    token_count: Optional[int] = None
    model_name: str = ""
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """Abstract base class for all LLM wrappers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize model wrapper
        
        Args:
            model_name: Name/identifier of the model
            api_key: API key for authentication
        """
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse containing generated text and metadata
        """
        pass

