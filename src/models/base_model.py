"""
Base class for all model wrappers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import utilities
from ..utils.api_helpers import get_rate_limiter, RateLimiter
from ..utils.config import load_config


@dataclass
class ModelResponse:
    """Container for model generation response"""
    text: str
    latency: float
    token_count: Optional[int] = None
    model_name: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResponse':
        """Create response from dictionary"""
        return cls(**data)


class BaseModel(ABC):
    """
    Abstract base class for all LLM wrappers
    
    Provides:
    - Rate limiting to prevent quota exhaustion
    - Configuration management
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize model wrapper
        
        Args:
            model_name: Name/identifier of the model
            api_key: API key for authentication
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.api_key = api_key
        
        # Load configuration
        try:
            config = load_config(config_path)
            self.config = config.get('api', {})
        except FileNotFoundError:
            # Use defaults if config not found
            self.config = {
                'max_retries': 3,
                'retry_delay': 1.0,
                'timeout': 30,
                'rate_limits': {}
            }
        
        # Get rate limiter for this provider
        provider = self._get_provider_name()
        rate_limits = self.config.get('rate_limits', {})
        min_delay = rate_limits.get(provider, 0.5)
        self.rate_limiter = get_rate_limiter(provider, min_delay)
    
    def _get_provider_name(self) -> str:
        """
        Get provider name for rate limiting
        
        Returns:
            Provider name (e.g., 'openai', 'anthropic')
        """
        # Default implementation - subclasses can override
        model_lower = self.model_name.lower()
        
        if 'gpt' in model_lower or 'openai' in model_lower:
            return 'openai'
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return 'anthropic'
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'google'
        elif 'llama' in model_lower and 'meta/' in self.model_name:
            return 'replicate'
        elif 'llama' in model_lower:
            return 'together'
        else:
            return 'default'
    
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
            
        Note:
            Subclasses should implement the actual API call logic.
            Rate limiting is handled by this base class.
        """
        pass

