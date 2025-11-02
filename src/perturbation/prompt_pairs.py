"""
Generate and manage prompt pairs
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptPair:
    """Container for a pair of semantically similar prompts"""
    prompt1: str
    prompt2: str
    distance: float
    category: str
    epsilon_target: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptPair':
        """Create from dictionary"""
        return cls(**data)


class PromptPairGenerator:
    """Generate pairs of semantically similar prompts"""
    
    def __init__(self, paraphrase_generator, semantic_filter):
        """
        Initialize prompt pair generator
        
        Args:
            paraphrase_generator: ParaphraseGenerator instance
            semantic_filter: SemanticFilter instance
        """
        self.paraphrase_generator = paraphrase_generator
        self.semantic_filter = semantic_filter
        logger.info("Initialized PromptPairGenerator")
    
    def generate_pairs(
        self,
        base_prompts: List[str],
        epsilon: float,
        n_pairs_per_prompt: int = 10,
        category: str = "general",
        tolerance: float = 0.01,
        n_paraphrases: int = 100,
        paraphrase_method: str = "batch"
    ) -> List[PromptPair]:
        """
        Generate prompt pairs at target epsilon distance
        
        Args:
            base_prompts: List of base prompts
            epsilon: Target semantic distance
            n_pairs_per_prompt: Number of pairs to generate per base prompt
            category: Category label for these prompts
            tolerance: Acceptable distance tolerance (epsilon ± tolerance)
            n_paraphrases: Number of paraphrases to generate per base prompt
            paraphrase_method: Method for generating paraphrases ('batch' or 'iterative')
            
        Returns:
            List of PromptPair objects
        """
        all_pairs = []
        
        logger.info(
            f"Generating pairs for {len(base_prompts)} base prompts "
            f"at ε={epsilon:.4f} ± {tolerance:.4f}"
        )
        
        for i, base_prompt in enumerate(base_prompts):
            logger.info(f"Processing base prompt {i+1}/{len(base_prompts)}")
            
            # Generate paraphrases
            paraphrases = self.paraphrase_generator.generate_paraphrases(
                prompt=base_prompt,
                n_paraphrases=n_paraphrases,
                method=paraphrase_method
            )
            
            if not paraphrases:
                logger.warning(f"No paraphrases generated for prompt {i+1}")
                continue
            
            # Filter by semantic distance
            filtered = self.semantic_filter.filter_by_distance(
                base_prompt=base_prompt,
                candidate_prompts=paraphrases,
                epsilon=epsilon,
                tolerance=tolerance,
                max_results=n_pairs_per_prompt
            )
            
            # Create PromptPair objects
            for paraphrase, distance in filtered:
                pair = PromptPair(
                    prompt1=base_prompt,
                    prompt2=paraphrase,
                    distance=distance,
                    category=category,
                    epsilon_target=epsilon
                )
                all_pairs.append(pair)
            
            logger.info(
                f"Generated {len(filtered)} pairs for prompt {i+1} "
                f"(target: {n_pairs_per_prompt})"
            )
        
        logger.info(f"Total pairs generated: {len(all_pairs)}")
        return all_pairs
    
    def generate_multi_epsilon_pairs(
        self,
        base_prompts: List[str],
        epsilon_levels: List[float],
        n_pairs_per_prompt: int = 10,
        category: str = "general",
        **kwargs
    ) -> Dict[float, List[PromptPair]]:
        """
        Generate prompt pairs at multiple epsilon levels
        
        Args:
            base_prompts: List of base prompts
            epsilon_levels: List of epsilon values to generate pairs for
            n_pairs_per_prompt: Number of pairs per prompt per epsilon
            category: Category label
            **kwargs: Additional arguments passed to generate_pairs
            
        Returns:
            Dictionary mapping epsilon -> list of PromptPair objects
        """
        results = {}
        
        for epsilon in epsilon_levels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating pairs for ε = {epsilon:.4f}")
            logger.info(f"{'='*60}")
            
            pairs = self.generate_pairs(
                base_prompts=base_prompts,
                epsilon=epsilon,
                n_pairs_per_prompt=n_pairs_per_prompt,
                category=category,
                **kwargs
            )
            
            results[epsilon] = pairs
        
        return results
    
    def save_pairs(
        self,
        pairs: List[PromptPair],
        output_path: Path,
        format: str = "json"
    ):
        """
        Save prompt pairs to file
        
        Args:
            pairs: List of PromptPair objects
            output_path: Path to save file
            format: Output format ('json' or 'jsonl')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Save as JSON array
            with open(output_path, 'w') as f:
                json.dump([pair.to_dict() for pair in pairs], f, indent=2)
        
        elif format == "jsonl":
            # Save as JSON lines
            with open(output_path, 'w') as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + '\n')
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved {len(pairs)} pairs to {output_path}")
    
    @staticmethod
    def load_pairs(
        input_path: Path,
        format: str = "auto"
    ) -> List[PromptPair]:
        """
        Load prompt pairs from file
        
        Args:
            input_path: Path to input file
            format: Input format ('json', 'jsonl', or 'auto' to detect)
            
        Returns:
            List of PromptPair objects
        """
        input_path = Path(input_path)
        
        # Auto-detect format
        if format == "auto":
            if input_path.suffix == ".jsonl":
                format = "jsonl"
            else:
                format = "json"
        
        pairs = []
        
        if format == "json":
            with open(input_path, 'r') as f:
                data = json.load(f)
                pairs = [PromptPair.from_dict(item) for item in data]
        
        elif format == "jsonl":
            with open(input_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    pairs.append(PromptPair.from_dict(data))
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Loaded {len(pairs)} pairs from {input_path}")
        return pairs
    
    def get_statistics(
        self,
        pairs: List[PromptPair]
    ) -> Dict:
        """
        Get statistics about generated prompt pairs
        
        Args:
            pairs: List of PromptPair objects
            
        Returns:
            Dictionary of statistics
        """
        import numpy as np
        
        if not pairs:
            return {}
        
        distances = [pair.distance for pair in pairs]
        epsilons = [pair.epsilon_target for pair in pairs]
        categories = [pair.category for pair in pairs]
        
        stats = {
            "n_pairs": len(pairs),
            "distance_stats": {
                "mean": float(np.mean(distances)),
                "std": float(np.std(distances)),
                "min": float(np.min(distances)),
                "max": float(np.max(distances)),
                "median": float(np.median(distances)),
            },
            "epsilon_targets": list(set(epsilons)),
            "categories": list(set(categories)),
            "pairs_per_category": {
                cat: sum(1 for p in pairs if p.category == cat)
                for cat in set(categories)
            },
            "pairs_per_epsilon": {
                eps: sum(1 for p in pairs if p.epsilon_target == eps)
                for eps in set(epsilons)
            }
        }
        
        return stats

