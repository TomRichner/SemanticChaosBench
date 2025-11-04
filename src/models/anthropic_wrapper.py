"""
Anthropic API wrapper
"""

import os
import time
from typing import Optional
from anthropic import Anthropic, APIError, APITimeoutError, RateLimitError, InternalServerError
from .base_model import BaseModel, ModelResponse
from ..utils.api_helpers import retry_with_backoff, RetryConfig


class AnthropicModel(BaseModel):
    """Wrapper for Anthropic models (Claude) with caching, retries, and rate limiting"""
    
    def __init__(
        self,
        model_name: str = "claude-haiku-4-5",
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        config_path: str = "config.yaml"
    ):
        """
        Initialize Anthropic model wrapper
        
        Args:
            model_name: Anthropic model name (e.g., 'claude-haiku-4-5')
            api_key: Anthropic API key
            enable_cache: Whether to enable response caching
            config_path: Path to configuration file
        """
        super().__init__(model_name, api_key, enable_cache, config_path)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY in .env")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
    
    def _get_provider_name(self) -> str:
        """Override to return 'anthropic'"""
        return 'anthropic'
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Anthropic API with caching and rate limiting
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        # Check cache first
        cache_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
        if system_prompt:
            cache_kwargs['system_prompt'] = system_prompt
        
        cached_response = self._get_cached_response(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **cache_kwargs
        )
        if cached_response:
            return cached_response
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API call with retry logic
        response = self._generate_with_retry(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Cache the response
        self._cache_response(
            response=response,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **cache_kwargs
        )
        
        return response
    
    def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Internal method to generate with retry logic
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        # Define retryable exceptions
        retryable_exceptions = (
            RateLimitError,
            APITimeoutError,
            InternalServerError,
            APIError
        )
        
        # Create retry configuration
        retry_config = RetryConfig(
            max_retries=self.config.get('max_retries', 3),
            base_delay=self.config.get('retry_delay', 1.0),
            exponential_backoff=True
        )
        
        # Apply retry decorator
        @retry_with_backoff(retryable_exceptions=retryable_exceptions, config=retry_config)
        def _make_api_call():
            start_time = time.time()
            
            # Filter out system_prompt from kwargs
            gen_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
            
            # Make API call
            # Anthropic uses system parameter separately
            api_kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                **gen_kwargs
            }
            
            if system_prompt:
                api_kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**api_kwargs)
            
            latency = time.time() - start_time
            
            # Extract response data
            generated_text = response.content[0].text
            
            # Calculate token count
            token_count = None
            if hasattr(response, 'usage'):
                token_count = response.usage.input_tokens + response.usage.output_tokens
            
            return ModelResponse(
                text=generated_text,
                latency=latency,
                token_count=token_count,
                model_name=self.model_name,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') else None,
                    "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') else None,
                }
            )
        
        try:
            return _make_api_call()
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {str(e)}")

