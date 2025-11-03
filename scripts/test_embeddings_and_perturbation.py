#!/usr/bin/env python3
"""
Test script for Sentence-BERT embeddings and prompt perturbation generator

This script tests:
1. Sentence-BERT embeddings with MPS acceleration
2. Paraphrase generation using LLM APIs
3. Semantic filtering by distance
4. Complete prompt pair generation pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_embeddings():
    """Test Sentence-BERT embeddings with MPS acceleration"""
    from measurement.embeddings import EmbeddingModel
    
    logger.info("\n" + "="*60)
    logger.info("Testing Sentence-BERT Embeddings")
    logger.info("="*60)
    
    # Initialize model (will auto-detect MPS)
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2", device="auto")
    logger.info(f"Model: {model}")
    
    # Test single encoding
    text = "What is the capital of France?"
    embedding = model.encode(text)
    logger.info(f"Single text embedding shape: {embedding.shape}")
    logger.info(f"Embedding dimension: {model.get_embedding_dim()}")
    
    # Test batch encoding
    texts = [
        "What is the capital of France?",
        "What's the capital city of France?",
        "Paris is the capital of France.",
        "The capital of Germany is Berlin."
    ]
    embeddings = model.encode(texts)
    logger.info(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test cosine distance
    dist_1_2 = model.cosine_distance(embeddings[0], embeddings[1])
    dist_1_3 = model.cosine_distance(embeddings[0], embeddings[2])
    dist_1_4 = model.cosine_distance(embeddings[0], embeddings[3])
    
    logger.info(f"Distance between similar questions: {dist_1_2:.4f}")
    logger.info(f"Distance to answer: {dist_1_3:.4f}")
    logger.info(f"Distance to different topic: {dist_1_4:.4f}")
    
    # Test pairwise distances
    pairwise = model.pairwise_distances(embeddings)
    logger.info(f"Pairwise distance matrix shape: {pairwise.shape}")
    
    logger.info("✓ Embeddings test passed!")
    return model


def test_paraphrase_generator(model_name="gemini-2.5-flash"):
    """Test paraphrase generation"""
    from perturbation.paraphrase_generator import ParaphraseGenerator
    
    logger.info("\n" + "="*60)
    logger.info(f"Testing Paraphrase Generator ({model_name})")
    logger.info("="*60)
    
    try:
        # Initialize generator
        generator = ParaphraseGenerator(model_name=model_name)
        
        # Test prompt
        base_prompt = "What is the capital of France?"
        
        # Generate paraphrases (small batch for testing)
        logger.info(f"Generating 5 paraphrases of: '{base_prompt}'")
        paraphrases = generator.generate_paraphrases(
            prompt=base_prompt,
            n_paraphrases=5,
            temperature=0.9,
            method="batch"
        )
        
        logger.info(f"Generated {len(paraphrases)} paraphrases:")
        for i, p in enumerate(paraphrases, 1):
            logger.info(f"  {i}. {p}")
        
        logger.info("✓ Paraphrase generator test passed!")
        return generator, paraphrases
        
    except Exception as e:
        logger.error(f"Paraphrase generator test failed: {e}")
        logger.warning("This may be due to missing API keys")
        return None, []


def test_semantic_filter(embedding_model):
    """Test semantic filtering"""
    from perturbation.semantic_filter import SemanticFilter
    
    logger.info("\n" + "="*60)
    logger.info("Testing Semantic Filter")
    logger.info("="*60)
    
    # Initialize filter
    filter_obj = SemanticFilter(embedding_model)
    
    # Test data
    base_prompt = "What is the capital of France?"
    candidates = [
        "What's the capital city of France?",  # Very similar
        "Can you tell me France's capital?",  # Similar
        "Paris is a beautiful city.",  # Less similar
        "What is the weather like today?",  # Different
    ]
    
    # Get distance distribution
    distances, sorted_prompts = filter_obj.get_distance_distribution(
        base_prompt, candidates
    )
    
    logger.info("Distance distribution:")
    for dist, prompt in zip(distances, sorted_prompts):
        logger.info(f"  {dist:.4f}: {prompt}")
    
    # Test filtering at different epsilon levels
    epsilon = 0.05
    tolerance = 0.02
    
    filtered = filter_obj.filter_by_distance(
        base_prompt=base_prompt,
        candidate_prompts=candidates,
        epsilon=epsilon,
        tolerance=tolerance
    )
    
    logger.info(f"\nFiltered at ε={epsilon:.4f} ± {tolerance:.4f}:")
    for prompt, dist in filtered:
        logger.info(f"  {dist:.4f}: {prompt}")
    
    # Test diversity score
    diversity = filter_obj.compute_diversity_score(candidates)
    logger.info(f"\nDiversity score: {diversity:.4f}")
    
    logger.info("✓ Semantic filter test passed!")
    return filter_obj


def test_prompt_pair_generator(embedding_model, use_api=False):
    """Test complete prompt pair generation pipeline"""
    from perturbation.prompt_pairs import PromptPairGenerator
    from perturbation.paraphrase_generator import ParaphraseGenerator
    from perturbation.semantic_filter import SemanticFilter
    
    logger.info("\n" + "="*60)
    logger.info("Testing Prompt Pair Generator")
    logger.info("="*60)
    
    if not use_api:
        logger.info("Skipping full pipeline test (no API)")
        logger.info("Set use_api=True to test with real API calls")
        return
    
    try:
        # Initialize components
        paraphrase_gen = ParaphraseGenerator(model_name="gemini-2.5-flash")
        semantic_filter = SemanticFilter(embedding_model)
        pair_gen = PromptPairGenerator(paraphrase_gen, semantic_filter)
        
        # Test prompts
        base_prompts = [
            "What is the capital of France?",
        ]
        
        # Generate pairs
        logger.info("Generating prompt pairs...")
        pairs = pair_gen.generate_pairs(
            base_prompts=base_prompts,
            epsilon=0.05,
            n_pairs_per_prompt=3,
            category="factual",
            tolerance=0.02,
            n_paraphrases=10,
            paraphrase_method="batch"
        )
        
        logger.info(f"\nGenerated {len(pairs)} prompt pairs:")
        for i, pair in enumerate(pairs, 1):
            logger.info(f"\nPair {i}:")
            logger.info(f"  Prompt 1: {pair.prompt1}")
            logger.info(f"  Prompt 2: {pair.prompt2}")
            logger.info(f"  Distance: {pair.distance:.4f}")
            logger.info(f"  Target ε: {pair.epsilon_target:.4f}")
        
        # Get statistics
        stats = pair_gen.get_statistics(pairs)
        logger.info(f"\nStatistics:")
        logger.info(f"  Total pairs: {stats['n_pairs']}")
        logger.info(f"  Mean distance: {stats['distance_stats']['mean']:.4f}")
        logger.info(f"  Std distance: {stats['distance_stats']['std']:.4f}")
        
        # Test saving/loading
        from pathlib import Path
        test_file = Path("data/prompt_pairs/test_pairs.json")
        pair_gen.save_pairs(pairs, test_file, format="json")
        
        loaded_pairs = pair_gen.load_pairs(test_file)
        logger.info(f"\nSaved and loaded {len(loaded_pairs)} pairs")
        
        logger.info("✓ Prompt pair generator test passed!")
        
    except Exception as e:
        logger.error(f"Prompt pair generator test failed: {e}")
        logger.warning("This may be due to missing API keys")


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("SEMANTIC CHAOS BENCH - Component Tests")
    logger.info("="*60)
    
    # Test 1: Embeddings (always runs, uses local model)
    embedding_model = test_embeddings()
    
    # Test 2: Semantic filter (uses embeddings, no API needed)
    test_semantic_filter(embedding_model)
    
    # Test 3: Paraphrase generator (requires API key)
    logger.info("\nTrying to test paraphrase generator...")
    paraphrase_gen, paraphrases = test_paraphrase_generator(model_name="gemini-2.5-flash")
    
    # Test 4: Full pipeline (requires API key)
    if paraphrase_gen is not None:
        test_prompt_pair_generator(embedding_model, use_api=True)
    else:
        logger.warning("\nSkipping full pipeline test (no API available)")
        logger.info("To test the full pipeline, ensure you have API keys set up in .env")
    
    logger.info("\n" + "="*60)
    logger.info("All tests completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

