"""
OpenAI API wrapper
"""

import os
import time
from typing import Optional
from openai import OpenAI
from .base_model import BaseModel, ModelResponse


class OpenAIModel(BaseModel):
    """Wrapper for OpenAI models"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize OpenAI model wrapper
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o-mini')
            api_key: OpenAI API key
        """
        super().__init__(model_name, api_key)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional OpenAI-specific parameters
            
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
            
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

