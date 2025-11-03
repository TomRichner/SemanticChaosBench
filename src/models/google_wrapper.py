"""
Google Gemini API wrapper (via Google AI Studio)
"""

import os
import time
from typing import Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from .base_model import BaseModel, ModelResponse


class GoogleModel(BaseModel):
    """Wrapper for Google models (Gemini via AI Studio)"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize Google model wrapper (AI Studio API)
        
        Args:
            model_name: Google model name (e.g., 'gemini-2.5-flash')
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
        max_tokens: int = 800,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using Google AI Studio API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
            system_prompt: Optional system prompt (will be prepended to user prompt)
            max_retries: Maximum number of retries for server errors (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
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
        # Filter out system_prompt and retry params from kwargs as they're not valid GenerationConfig parameters
        gen_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ('system_prompt', 'max_retries', 'retry_delay')}
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **gen_kwargs
        )
        
        # Retry loop for handling transient server errors
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Configure safety settings to be less restrictive
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]
                
                # Generate content
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                latency = time.time() - start_time
                
                # Check if response was blocked
                if not response.candidates:
                    raise RuntimeError("Response blocked by Google AI. No candidates returned.")
                
                candidate = response.candidates[0]
                
                # Try to extract text - if it fails, check why
                try:
                    generated_text = response.text
                except ValueError as e:
                    # response.text raises ValueError when there are no valid parts
                    # Check the finish reason to give a better error message
                    finish_reason = candidate.finish_reason
                    reason_name = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    
                    # Include safety ratings if available
                    safety_ratings = ""
                    if hasattr(candidate, 'safety_ratings'):
                        ratings = candidate.safety_ratings  
                        if ratings:
                            safety_ratings = f"\nSafety ratings: {[f'{r.category.name}={r.probability.name}' for r in ratings]}"
                    
                    # Different messages for different finish reasons
                    if reason_name == 'MAX_TOKENS':
                        raise RuntimeError(
                            f"Google AI returned no content (MAX_TOKENS reached before any output was generated). "
                            f"Try increasing max_tokens (current: {max_tokens}) or shortening the prompt."
                        )
                    elif reason_name in ['SAFETY', 'RECITATION']:
                        raise RuntimeError(
                            f"Response blocked by Google AI due to {reason_name} filters.{safety_ratings}"
                        )
                    else:
                        raise RuntimeError(
                            f"Response blocked by Google AI. Finish reason: {reason_name}{safety_ratings}\nOriginal error: {str(e)}"
                        )
                
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
                        "attempts": attempt + 1,
                    }
                )
                
            except RuntimeError:
                # Re-raise RuntimeError as-is (already has good error message)
                # These are typically non-retryable errors (safety blocks, etc.)
                raise
            except google_exceptions.InternalServerError as e:
                # Google internal server error (500) - retryable
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
            except google_exceptions.ServiceUnavailable as e:
                # Service temporarily unavailable (503) - retryable
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
            except google_exceptions.ResourceExhausted as e:
                # Rate limit or quota exceeded (429) - retryable with longer delay
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1) * 2)  # Longer backoff for rate limits
                    continue
            except Exception as e:
                # Handle other API errors - might be retryable
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise RuntimeError(f"Google AI generation failed after {max_retries} attempts: {str(last_exception)}")

