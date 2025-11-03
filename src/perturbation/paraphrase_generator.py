"""
Generate paraphrased prompts using LLM APIs
"""

from typing import List, Optional
import logging

# Lazy imports to avoid circular dependencies
def _get_model(model_name: str, api_key: Optional[str] = None):
    """Dynamically import and return the appropriate model wrapper"""
    if "gpt" in model_name.lower():
        from src.models.openai_wrapper import OpenAIModel
        return OpenAIModel(model_name=model_name, api_key=api_key)
    elif "claude" in model_name.lower():
        from src.models.anthropic_wrapper import AnthropicModel
        return AnthropicModel(model_name=model_name, api_key=api_key)
    elif "gemini" in model_name.lower():
        from src.models.google_wrapper import GoogleModel
        return GoogleModel(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

logger = logging.getLogger(__name__)


class ParaphraseGenerator:
    """Generate paraphrases of prompts using LLMs"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize paraphrase generator
        
        Args:
            model_name: Name of model to use for paraphrasing
                       Supported: gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash
            api_key: Optional API key (if not provided, will use environment variable)
        """
        self.model_name = model_name
        self.model = _get_model(model_name, api_key)
        logger.info(f"Initialized ParaphraseGenerator with model: {model_name}")
    
    def generate_paraphrases(
        self,
        prompt: str,
        n_paraphrases: int = 100,
        temperature: float = 0.9,
        method: str = "batch"
    ) -> List[str]:
        """
        Generate multiple paraphrases of a prompt
        
        Args:
            prompt: Original prompt
            n_paraphrases: Number of paraphrases to generate
            temperature: Generation temperature (higher = more variation)
            method: Generation method ('batch' or 'iterative')
                   - 'batch': Generate all paraphrases in one API call
                   - 'iterative': Generate paraphrases one at a time
            
        Returns:
            List of paraphrased prompts
        """
        logger.info(f"Generating {n_paraphrases} paraphrases for prompt using {method} method")
        
        if method == "batch":
            return self._generate_batch(prompt, n_paraphrases, temperature)
        elif method == "iterative":
            return self._generate_iterative(prompt, n_paraphrases, temperature)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_batch(
        self,
        prompt: str,
        n_paraphrases: int,
        temperature: float
    ) -> List[str]:
        """
        Generate paraphrases in a single batch API call
        
        This is more efficient but may have less variation between paraphrases
        """
        system_prompt = """You are a helpful assistant that generates paraphrases.
Your task is to create multiple variations of a given prompt that preserve the original meaning
while using different wording, structure, and phrasing.

Guidelines:
- Maintain the core meaning and intent
- Vary sentence structure and word choice
- Keep similar length (±20%)
- Use natural, fluent language
- Avoid trivial changes (e.g., only changing punctuation)
- Create diverse variations"""

        user_prompt = f"""Generate {n_paraphrases} diverse paraphrases of the following prompt.
Return ONLY the paraphrases, one per line, without numbering or bullet points.

Original prompt:
{prompt}

Paraphrases:"""

        # Generate paraphrases
        response = self.model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max(2000, n_paraphrases * 50)  # Estimate tokens needed
        )
        
        # Parse response into list of paraphrases
        paraphrases = [
            line.strip()
            for line in response.strip().split('\n')
            if line.strip() and not line.strip().startswith(('#', '-', '*', str(i)))
            for i in range(n_paraphrases + 1)
        ]
        
        # Clean up: remove numbering if present
        cleaned_paraphrases = []
        for p in paraphrases:
            # Remove leading numbers like "1.", "2)", etc.
            import re
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', p)
            if cleaned and len(cleaned) > 10:  # Filter out very short responses
                cleaned_paraphrases.append(cleaned)
        
        logger.info(f"Generated {len(cleaned_paraphrases)} paraphrases via batch method")
        return cleaned_paraphrases[:n_paraphrases]
    
    def _generate_iterative(
        self,
        prompt: str,
        n_paraphrases: int,
        temperature: float
    ) -> List[str]:
        """
        Generate paraphrases one at a time
        
        This is slower but provides more variation and independence between paraphrases
        """
        system_prompt = """You are a helpful assistant that generates paraphrases.
Create a single variation of the given prompt that preserves the original meaning
while using different wording, structure, and phrasing."""

        paraphrases = []
        
        for i in range(n_paraphrases):
            user_prompt = f"""Paraphrase the following prompt while maintaining its meaning.
Return ONLY the paraphrase, nothing else.

Original prompt:
{prompt}

Paraphrase:"""

            # Generate single paraphrase
            response = self.model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=500
            )
            
            paraphrase = response.strip()
            if paraphrase and len(paraphrase) > 10:
                paraphrases.append(paraphrase)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_paraphrases} paraphrases")
        
        logger.info(f"Generated {len(paraphrases)} paraphrases via iterative method")
        return paraphrases
    
    def generate_with_temperature_variation(
        self,
        prompt: str,
        n_variations: int = 50,
        temperature_range: tuple = (1.2, 1.8)
    ) -> List[str]:
        """
        Alternative method: Generate variations using high temperature
        
        This method uses the model to slightly rephrase the prompt using
        high temperature to introduce natural variation.
        
        Args:
            prompt: Original prompt
            n_variations: Number of variations to generate
            temperature_range: (min_temp, max_temp) for generation
            
        Returns:
            List of prompt variations
        """
        import numpy as np
        
        system_prompt = """You are helping create slight variations of prompts.
Rephrase the given prompt with minor wording changes while keeping the exact same meaning."""

        variations = []
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_variations)
        
        for temp in temperatures:
            user_prompt = f"""Create a slight variation of this prompt (keep the same meaning):

{prompt}

Variation:"""

            response = self.model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=float(temp),
                max_tokens=500
            )
            
            variation = response.strip()
            if variation and len(variation) > 10:
                variations.append(variation)
        
        logger.info(f"Generated {len(variations)} variations via temperature method")
        return variations

