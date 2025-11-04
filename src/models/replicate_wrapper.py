"""
Replicate API wrapper
"""

import os
import time
from typing import Optional
import replicate
from .base_model import BaseModel, ModelResponse
from ..utils.api_helpers import retry_with_backoff, RetryConfig


class ReplicateModel(BaseModel):
    """Wrapper for Replicate models (open source models) with caching, retries, and rate limiting"""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        enable_cache: bool = False,
        config_path: str = "config.yaml"
    ):
        """
        Initialize Replicate model wrapper
        
        Args:
            model_name: Replicate model identifier (e.g., 'meta/meta-llama-3-8b-instruct')
            api_key: Replicate API token
            enable_cache: Whether to enable response caching
            config_path: Path to configuration file
        """
        super().__init__(model_name, api_key, enable_cache, config_path)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("Replicate API token required. Set REPLICATE_API_TOKEN in .env")
        
        # Set API token for replicate library
        os.environ["REPLICATE_API_TOKEN"] = api_key
        self.client = replicate.Client(api_token=api_key)
    
    def _get_provider_name(self) -> str:
        """Override to return 'replicate'"""
        return 'replicate'
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Replicate API with caching and rate limiting
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional Replicate-specific parameters
            
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
            **kwargs: Additional Replicate-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        # Define retryable exceptions (Replicate uses general exceptions)
        retryable_exceptions = (Exception,)
        
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
            
            # Build input parameters
            input_params = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # Add system prompt if provided
            if system_prompt:
                input_params["system_prompt"] = system_prompt
            
            # Run prediction
            output = self.client.run(
                self.model_name,
                input=input_params
            )
            
            latency = time.time() - start_time
            
            # Replicate returns different formats depending on model
            # Most text models return iterator or list of strings
            if isinstance(output, str):
                generated_text = output
            elif hasattr(output, '__iter__'):
                # Join iterator/list outputs
                generated_text = "".join(str(chunk) for chunk in output)
            else:
                generated_text = str(output)
            
            return ModelResponse(
                text=generated_text,
                latency=latency,
                token_count=None,  # Replicate doesn't always provide token counts
                model_name=self.model_name,
                metadata={}
            )
        
        try:
            return _make_api_call()
        except Exception as e:
            raise RuntimeError(f"Replicate generation failed: {str(e)}")

