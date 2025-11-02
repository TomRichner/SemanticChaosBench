"""
Statistical metrics for divergence analysis
"""

import numpy as np
from typing import List, Dict, Any


class MetricsCalculator:
    """Calculate divergence metrics and Lyapunov-like exponents"""
    
    def compute_lyapunov_exponent(
        self,
        divergence_trajectory: List[float],
        initial_divergence: float,
        time_steps: List[int]
    ) -> float:
        """
        Compute Lyapunov-like exponent: λ ≈ (1/t) * log(δ(t)/δ(0))
        
        Args:
            divergence_trajectory: List of divergence values over time
            initial_divergence: Initial divergence δ(0)
            time_steps: Corresponding time steps
            
        Returns:
            Estimated Lyapunov exponent
        """
        # TODO: Implement Lyapunov exponent calculation
        raise NotImplementedError("Lyapunov exponent calculation not yet implemented")
    
    def compute_saturation_distance(
        self,
        divergence_trajectory: List[float]
    ) -> float:
        """
        Compute saturation distance (maximum divergence reached)
        
        Args:
            divergence_trajectory: List of divergence values
            
        Returns:
            Maximum divergence
        """
        return max(divergence_trajectory) if divergence_trajectory else 0.0
    
    def compute_divergence_onset(
        self,
        divergence_trajectory: List[float],
        threshold: float = 0.1
    ) -> int:
        """
        Compute divergence onset (steps until significant divergence)
        
        Args:
            divergence_trajectory: List of divergence values
            threshold: Threshold for significant divergence
            
        Returns:
            Step number where divergence exceeds threshold
        """
        for i, div in enumerate(divergence_trajectory):
            if div > threshold:
                return i
        return len(divergence_trajectory)

