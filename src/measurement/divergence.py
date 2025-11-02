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
            Dictionary with divergence metrics
        """
        # TODO: Implement single-step divergence measurement
        raise NotImplementedError("Single divergence measurement not yet implemented")
    
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

