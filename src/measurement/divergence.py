"""
Divergence measurement between model outputs
"""

from typing import Dict, Any
from .embeddings import EmbeddingModel


class DivergenceMeasurer:
    """Measure semantic divergence between model outputs"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize divergence measurer
        
        Args:
            embedding_model: Model for computing embeddings
        """
        self.embedding_model = embedding_model
    
    def measure_single_divergence(
        self,
        prompt1: str,
        prompt2: str,
        output1: str,
        output2: str
    ) -> Dict[str, Any]:
        """
        Measure single-step divergence
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            output1: Output from prompt1
            output2: Output from prompt2
            
        Returns:
            Dictionary with divergence metrics:
                - input_distance: Semantic distance between prompts
                - output_distance: Semantic distance between outputs
                - divergence_rate: output_distance / input_distance
                - prompt_embeddings: (embedding1, embedding2)
                - output_embeddings: (embedding1, embedding2)
        """
        # Embed the prompts
        prompt_embeddings = self.embedding_model.encode([prompt1, prompt2])
        prompt_emb1, prompt_emb2 = prompt_embeddings[0], prompt_embeddings[1]
        
        # Embed the outputs
        output_embeddings = self.embedding_model.encode([output1, output2])
        output_emb1, output_emb2 = output_embeddings[0], output_embeddings[1]
        
        # Compute distances
        input_distance = self.embedding_model.cosine_distance(prompt_emb1, prompt_emb2)
        output_distance = self.embedding_model.cosine_distance(output_emb1, output_emb2)
        
        # Compute divergence rate
        divergence_rate = self.compute_divergence_rate(input_distance, output_distance)
        
        return {
            'input_distance': float(input_distance),
            'output_distance': float(output_distance),
            'divergence_rate': float(divergence_rate),
            'prompt_embeddings': (prompt_emb1, prompt_emb2),
            'output_embeddings': (output_emb1, output_emb2),
        }
    
    def compute_divergence_rate(
        self,
        input_distance: float,
        output_distance: float
    ) -> float:
        """
        Compute divergence rate δ(t) = ||output1 - output2|| / ||prompt1 - prompt2||
        
        Args:
            input_distance: Semantic distance between inputs
            output_distance: Semantic distance between outputs
            
        Returns:
            Divergence rate
        """
        if input_distance == 0:
            return float('inf')
        return output_distance / input_distance

