"""
Replicate API wrapper
"""

import os
import time
from typing import Optional
import replicate
from .base_model import BaseModel, ModelResponse


class ReplicateModel(BaseModel):
    """Wrapper for Replicate models (open source models)"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize Replicate model wrapper
        
        Args:
            model_name: Replicate model identifier (e.g., 'meta/llama-3-8b')
            api_key: Replicate API token
        """
        super().__init__(model_name, api_key)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("Replicate API token required. Set REPLICATE_API_TOKEN in .env")
        
        # Set API token for replicate library
        os.environ["REPLICATE_API_TOKEN"] = api_key
        self.client = replicate.Client(api_token=api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Replicate API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional Replicate-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
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
        
        try:
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
            
        except Exception as e:
            raise RuntimeError(f"Replicate generation failed: {str(e)}")

