"""
OpenAI API wrapper
"""

import os
import time
from typing import Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, InternalServerError
from .base_model import BaseModel, ModelResponse
from ..utils.api_helpers import retry_with_backoff, RetryConfig


class OpenAIModel(BaseModel):
    """Wrapper for OpenAI models with caching, retries, and rate limiting"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        enable_cache: bool = False,
        config_path: str = "config.yaml"
    ):
        """
        Initialize OpenAI model wrapper
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o-mini')
            api_key: OpenAI API key
            enable_cache: Whether to enable response caching
            config_path: Path to configuration file
        """
        super().__init__(model_name, api_key, enable_cache, config_path)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def _get_provider_name(self) -> str:
        """Override to return 'openai'"""
        return 'openai'
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using OpenAI API with caching and rate limiting
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional OpenAI-specific parameters
            
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
            **kwargs: Additional OpenAI-specific parameters
            
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
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Filter out system_prompt from kwargs
            gen_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
            
            # Make API call
            # GPT-5 models use max_completion_tokens instead of max_tokens
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                **gen_kwargs
            }
            
            if "gpt-5" in self.model_name.lower() or "o1" in self.model_name.lower() or "o3" in self.model_name.lower():
                api_kwargs["max_completion_tokens"] = max_tokens
            else:
                api_kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**api_kwargs)
            
            latency = time.time() - start_time
            
            # Extract response data
            generated_text = response.choices[0].message.content
            token_count = response.usage.total_tokens if response.usage else None
            
            return ModelResponse(
                text=generated_text,
                latency=latency,
                token_count=token_count,
                model_name=self.model_name,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                }
            )
        
        try:
            return _make_api_call()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

