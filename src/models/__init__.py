"""
Unified interface for various LLM APIs
"""

from typing import Optional
from .base_model import BaseModel, ModelResponse
from .openai_wrapper import OpenAIModel
from .anthropic_wrapper import AnthropicModel
from .google_wrapper import GoogleModel
from .replicate_wrapper import ReplicateModel
from .together_wrapper import TogetherModel


def get_model(model_name: str, api_key: Optional[str] = None) -> BaseModel:
    """
    Factory function to get appropriate model wrapper based on model name.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-4o-mini', 'claude-haiku-4-5', 'gemini-2.5-flash')
        api_key: Optional API key (if not provided, will use environment variables)
        
    Returns:
        Initialized model wrapper instance
        
    Raises:
        ValueError: If model name is not recognized
        
    Examples:
        >>> model = get_model('gpt-4o-mini')
        >>> response = model.generate("Hello, world!")
        
        >>> model = get_model('claude-haiku-4-5')
        >>> response = model.generate("Explain quantum computing", temperature=0.5)
    """
    model_lower = model_name.lower()
    
    # OpenAI models
    if any(prefix in model_lower for prefix in ['gpt-', 'o1-', 'o3-']):
        return OpenAIModel(model_name=model_name, api_key=api_key)
    
    # Anthropic models
    elif 'claude' in model_lower:
        return AnthropicModel(model_name=model_name, api_key=api_key)
    
    # Google models
    elif 'gemini' in model_lower or 'palm' in model_lower:
        return GoogleModel(model_name=model_name, api_key=api_key)
    
    # Replicate models (typically have '/' in name like 'meta/llama-3-8b')
    elif '/' in model_name and any(prefix in model_lower for prefix in ['meta/', 'mistralai/', 'nousresearch/']):
        return ReplicateModel(model_name=model_name, api_key=api_key)
    
    # Together AI models
    elif 'mixtral' in model_lower or 'together' in model_lower or 'meta-llama/' in model_name:
        return TogetherModel(model_name=model_name, api_key=api_key)
    
    else:
        raise ValueError(
            f"Model '{model_name}' not recognized. Supported prefixes:\n"
            "  - OpenAI: gpt-, o1-, o3-\n"
            "  - Anthropic: claude\n"
            "  - Google: gemini, palm\n"
            "  - Replicate: meta/, mistralai/, etc.\n"
            "  - Together: mixtral, together"
        )


class ModelInterface:
    """
    Unified interface for LLM generation across multiple providers.
    
    This class provides a consistent API for generating text from various LLM providers
    (OpenAI, Anthropic, Google, Replicate, Together AI).
    
    Usage:
        >>> interface = ModelInterface()
        >>> response = interface.generate(
        ...     prompt="Explain quantum computing",
        ...     model="gpt-4o-mini",
        ...     temperature=0.7,
        ...     max_tokens=500
        ... )
        >>> print(f"Generated: {response.text}")
        >>> print(f"Latency: {response.latency:.2f}s")
        >>> print(f"Tokens: {response.token_count}")
    """
    
    def __init__(self):
        """Initialize the unified model interface."""
        self._model_cache = {}
    
    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text from a prompt using specified model.
        
        Args:
            prompt: Input prompt text
            model: Model identifier (e.g., 'gpt-4o-mini', 'claude-haiku-4-5', 'gemini-2.5-flash')
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            api_key: Optional API key (uses environment variables if not provided)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse containing:
                - text: Generated text
                - latency: Generation time in seconds
                - token_count: Total tokens used (if available)
                - model_name: Name of the model used
                - metadata: Additional model-specific metadata
                
        Raises:
            ValueError: If model is not recognized
            RuntimeError: If API call fails
        """
        # Get or create model wrapper
        cache_key = f"{model}:{api_key}" if api_key else model
        
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = get_model(model, api_key)
        
        model_wrapper = self._model_cache[cache_key]
        
        # Generate response
        return model_wrapper.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def clear_cache(self):
        """Clear the internal model cache."""
        self._model_cache.clear()


__all__ = [
    "BaseModel",
    "ModelResponse",
    "OpenAIModel",
    "AnthropicModel",
    "GoogleModel",
    "ReplicateModel",
    "TogetherModel",
    "ModelInterface",
    "get_model",
]

