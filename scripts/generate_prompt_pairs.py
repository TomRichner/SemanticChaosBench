#!/usr/bin/env python3
"""
Generate prompt pairs at various semantic distances

This script generates prompt pairs for the Semantic Chaos Bench benchmark.
It uses LLM APIs to generate paraphrases and filters them by semantic distance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import yaml
from dotenv import load_dotenv
from typing import List, Dict

from src.measurement.embeddings import EmbeddingModel
from src.perturbation.paraphrase_generator import ParaphraseGenerator
from src.perturbation.semantic_filter import SemanticFilter
from src.perturbation.prompt_pairs import PromptPairGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Example base prompts by category
EXAMPLE_PROMPTS = {
    "factual": [
        "What is the capital of France?",
        "When did World War II end?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "How many planets are in our solar system?",
    ],
    "creative": [
        "Write a short story about a time traveler.",
        "Describe a sunset over the ocean.",
        "Create a character for a fantasy novel.",
        "Compose a haiku about autumn.",
        "Imagine a world where gravity works in reverse.",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "What comes next in this sequence: 2, 4, 8, 16, 32, ?",
        "How would you explain why the sky is blue to a 5-year-old?",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Explain the difference between a list and a tuple in Python.",
        "How do you check if a number is prime?",
        "What is recursion and give an example?",
        "Write SQL to find duplicate records in a table.",
    ],
}


def load_config(config_path: Path = None) -> Dict:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def generate_pairs_for_category(
    category: str,
    prompts: List[str],
    epsilon_levels: List[float],
    pair_generator: PromptPairGenerator,
    output_dir: Path,
    n_pairs_per_prompt: int = 10,
    n_paraphrases: int = 100,
    tolerance: float = 0.01
):
    """Generate prompt pairs for a specific category"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating pairs for category: {category}")
    logger.info(f"{'='*60}")
    
    # Generate pairs at multiple epsilon levels
    results = pair_generator.generate_multi_epsilon_pairs(
        base_prompts=prompts,
        epsilon_levels=epsilon_levels,
        n_pairs_per_prompt=n_pairs_per_prompt,
        category=category,
        tolerance=tolerance,
        n_paraphrases=n_paraphrases,
        paraphrase_method="batch"
    )
    
    # Save results for each epsilon level
    for epsilon, pairs in results.items():
        if not pairs:
            logger.warning(f"No pairs generated for ε={epsilon:.4f}")
            continue
        
        # Save pairs
        output_file = output_dir / f"{category}_epsilon_{epsilon:.4f}.json"
        pair_generator.save_pairs(pairs, output_file, format="json")
        
        # Print statistics
        stats = pair_generator.get_statistics(pairs)
        logger.info(f"\nStatistics for ε={epsilon:.4f}:")
        logger.info(f"  Total pairs: {stats['n_pairs']}")
        logger.info(f"  Mean distance: {stats['distance_stats']['mean']:.4f}")
        logger.info(f"  Std distance: {stats['distance_stats']['std']:.4f}")
        logger.info(f"  Distance range: [{stats['distance_stats']['min']:.4f}, {stats['distance_stats']['max']:.4f}]")


def main():
    """Generate prompt pairs"""
    logger.info("="*60)
    logger.info("SEMANTIC CHAOS BENCH - Prompt Pair Generation")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    
    # Extract parameters from config
    epsilon_levels = config.get('epsilon_levels', [0.01, 0.05, 0.10, 0.20])
    paraphrase_model = config.get('paraphrase_model', 'gemini-2.5-flash')
    embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
    n_pairs_per_prompt = config.get('n_pairs_per_prompt', 10)
    n_paraphrases = config.get('n_paraphrases', 100)
    tolerance = config.get('tolerance', 0.01)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Epsilon levels: {epsilon_levels}")
    logger.info(f"  Paraphrase model: {paraphrase_model}")
    logger.info(f"  Embedding model: {embedding_model_name}")
    logger.info(f"  Pairs per prompt: {n_pairs_per_prompt}")
    logger.info(f"  Paraphrases to generate: {n_paraphrases}")
    logger.info(f"  Tolerance: ±{tolerance}")
    
    # Initialize components
    logger.info("\nInitializing components...")
    
    # 1. Embedding model (local, MPS-accelerated)
    embedding_model = EmbeddingModel(model_name=embedding_model_name, device="auto")
    
    # 2. Paraphrase generator (uses API)
    paraphrase_gen = ParaphraseGenerator(model_name=paraphrase_model)
    
    # 3. Semantic filter
    semantic_filter = SemanticFilter(embedding_model)
    
    # 4. Prompt pair generator
    pair_generator = PromptPairGenerator(paraphrase_gen, semantic_filter)
    
    # Set up output directory
    output_dir = Path(__file__).parent.parent / "data" / "prompt_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate pairs for each category
    categories_to_process = config.get('categories', list(EXAMPLE_PROMPTS.keys()))
    
    for category in categories_to_process:
        if category not in EXAMPLE_PROMPTS:
            logger.warning(f"Category '{category}' not found in examples, skipping")
            continue
        
        prompts = EXAMPLE_PROMPTS[category]
        
        try:
            generate_pairs_for_category(
                category=category,
                prompts=prompts,
                epsilon_levels=epsilon_levels,
                pair_generator=pair_generator,
                output_dir=output_dir,
                n_pairs_per_prompt=n_pairs_per_prompt,
                n_paraphrases=n_paraphrases,
                tolerance=tolerance
            )
        except Exception as e:
            logger.error(f"Error generating pairs for category '{category}': {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Prompt pair generation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
