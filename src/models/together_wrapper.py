"""
Together AI API wrapper
"""

import os
import time
from typing import Optional
from together import Together
from .base_model import BaseModel, ModelResponse


class TogetherModel(BaseModel):
    """Wrapper for Together AI models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize Together AI model wrapper
        
        Args:
            model_name: Together AI model identifier (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct-Lite')
            api_key: Together AI API key
        """
        super().__init__(model_name, api_key)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Together AI API key required. Set TOGETHER_API_KEY in .env")
        
        # Initialize Together client
        self.client = Together(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Together AI API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional Together-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Filter out system_prompt from kwargs
        gen_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **gen_kwargs
            )
            
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
            
        except Exception as e:
            raise RuntimeError(f"Together AI generation failed: {str(e)}")

