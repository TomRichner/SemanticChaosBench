"""
Filter prompt pairs by semantic distance
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticFilter:
    """Filter prompt pairs based on semantic distance"""
    
    def __init__(self, embedding_model):
        """
        Initialize semantic filter
        
        Args:
            embedding_model: EmbeddingModel instance for computing embeddings
        """
        self.embedding_model = embedding_model
        logger.info("Initialized SemanticFilter")
    
    def filter_by_distance(
        self,
        base_prompt: str,
        candidate_prompts: List[str],
        epsilon: float,
        tolerance: float = 0.01,
        max_results: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Filter prompts by semantic distance from base prompt
        
        Args:
            base_prompt: Original prompt
            candidate_prompts: List of candidate paraphrases
            epsilon: Target semantic distance
            tolerance: Acceptable distance tolerance (epsilon ± tolerance)
            max_results: Maximum number of results to return (None = all)
            
        Returns:
            List of (prompt, distance) pairs within epsilon ± tolerance,
            sorted by how close they are to epsilon
        """
        if not candidate_prompts:
            logger.warning("No candidate prompts provided")
            return []
        
        # Encode base prompt
        logger.info(f"Encoding base prompt and {len(candidate_prompts)} candidates")
        base_embedding = self.embedding_model.encode(base_prompt)
        
        # Encode all candidate prompts
        candidate_embeddings = self.embedding_model.encode(candidate_prompts)
        
        # Compute distances from base prompt to all candidates
        distances = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            distance = self.embedding_model.cosine_distance(
                base_embedding[0] if len(base_embedding.shape) > 1 else base_embedding,
                candidate_embedding
            )
            distances.append((candidate_prompts[i], float(distance)))
        
        # Filter by epsilon ± tolerance
        min_distance = epsilon - tolerance
        max_distance = epsilon + tolerance
        
        filtered = [
            (prompt, dist)
            for prompt, dist in distances
            if min_distance <= dist <= max_distance
        ]
        
        # Sort by how close to epsilon (prioritize exact matches)
        filtered_sorted = sorted(
            filtered,
            key=lambda x: abs(x[1] - epsilon)
        )
        
        # Limit results if requested
        if max_results is not None:
            filtered_sorted = filtered_sorted[:max_results]
        
        logger.info(
            f"Filtered {len(candidate_prompts)} candidates to {len(filtered_sorted)} "
            f"within distance range [{min_distance:.4f}, {max_distance:.4f}]"
        )
        
        return filtered_sorted
    
    def get_distance_distribution(
        self,
        base_prompt: str,
        candidate_prompts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distribution of distances from base prompt to candidates
        
        Useful for understanding the spread of paraphrases and choosing epsilon values
        
        Args:
            base_prompt: Original prompt
            candidate_prompts: List of candidate paraphrases
            
        Returns:
            Tuple of (distances, prompts) sorted by distance
        """
        # Encode base prompt
        base_embedding = self.embedding_model.encode(base_prompt)
        
        # Encode all candidate prompts
        candidate_embeddings = self.embedding_model.encode(candidate_prompts)
        
        # Compute distances
        distances = []
        for candidate_embedding in candidate_embeddings:
            distance = self.embedding_model.cosine_distance(
                base_embedding[0] if len(base_embedding.shape) > 1 else base_embedding,
                candidate_embedding
            )
            distances.append(distance)
        
        distances = np.array(distances)
        prompts = np.array(candidate_prompts)
        
        # Sort by distance
        sort_idx = np.argsort(distances)
        
        return distances[sort_idx], prompts[sort_idx]
    
    def find_optimal_epsilon_ranges(
        self,
        base_prompt: str,
        candidate_prompts: List[str],
        n_ranges: int = 4,
        min_samples_per_range: int = 5
    ) -> List[Tuple[float, float, int]]:
        """
        Identify optimal epsilon ranges that have sufficient candidate prompts
        
        This helps determine good epsilon values for generating prompt pairs
        
        Args:
            base_prompt: Original prompt
            candidate_prompts: List of candidate paraphrases
            n_ranges: Number of distance ranges to identify
            min_samples_per_range: Minimum number of prompts needed per range
            
        Returns:
            List of (epsilon_center, range_width, n_samples) tuples
        """
        distances, _ = self.get_distance_distribution(base_prompt, candidate_prompts)
        
        if len(distances) == 0:
            return []
        
        # Divide the distance space into ranges
        min_dist = distances.min()
        max_dist = distances.max()
        
        ranges = []
        range_edges = np.linspace(min_dist, max_dist, n_ranges + 1)
        
        for i in range(n_ranges):
            lower = range_edges[i]
            upper = range_edges[i + 1]
            
            # Count samples in this range
            n_samples = np.sum((distances >= lower) & (distances <= upper))
            
            if n_samples >= min_samples_per_range:
                epsilon_center = (lower + upper) / 2
                range_width = upper - lower
                ranges.append((epsilon_center, range_width, int(n_samples)))
        
        logger.info(f"Identified {len(ranges)} optimal epsilon ranges")
        for eps, width, n in ranges:
            logger.info(f"  ε={eps:.4f} ± {width/2:.4f} ({n} samples)")
        
        return ranges
    
    def compute_diversity_score(
        self,
        prompts: List[str]
    ) -> float:
        """
        Compute diversity score for a set of prompts
        
        Higher score = more diverse prompts
        
        Args:
            prompts: List of prompts
            
        Returns:
            Mean pairwise distance (diversity score)
        """
        if len(prompts) < 2:
            return 0.0
        
        # Encode all prompts
        embeddings = self.embedding_model.encode(prompts)
        
        # Compute pairwise distances
        distances = self.embedding_model.pairwise_distances(embeddings)
        
        # Get mean distance (excluding diagonal)
        n = len(prompts)
        diversity = (distances.sum() - n) / (n * (n - 1))
        
        return float(diversity)

