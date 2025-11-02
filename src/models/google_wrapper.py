"""
Google Gemini API wrapper (via Google AI Studio)
"""

import os
import time
from typing import Optional
import google.generativeai as genai
from .base_model import BaseModel, ModelResponse


class GoogleModel(BaseModel):
    """Wrapper for Google models (Gemini via AI Studio)"""
    
    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        """
        Initialize Google model wrapper (AI Studio API)
        
        Args:
            model_name: Google model name (e.g., 'gemini-2.5-pro', 'gemini-2.5-flash')
            api_key: Google API key from AI Studio
        """
        super().__init__(model_name, api_key)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY in .env")
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Google AI Studio API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
            system_prompt: Optional system prompt (will be prepended to user prompt)
            **kwargs: Additional Google-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Combine system prompt with user prompt if provided
        # Gemini doesn't have a separate system role in simple API
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Configure generation parameters
        # Filter out system_prompt from kwargs as it's not a valid GenerationConfig parameter
        gen_kwargs = {k: v for k, v in kwargs.items() if k != 'system_prompt'}
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **gen_kwargs
        )
        
        try:
            # Generate content
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            latency = time.time() - start_time
            
            # Extract text from response
            generated_text = response.text
            
            # Extract token count if available
            token_count = None
            if hasattr(response, 'usage_metadata'):
                token_count = response.usage_metadata.total_token_count
            
            return ModelResponse(
                text=generated_text,
                latency=latency,
                token_count=token_count,
                model_name=self.model_name,
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                }
            )
            
        except Exception as e:
            # Handle API errors
            raise RuntimeError(f"Google AI generation failed: {str(e)}")

