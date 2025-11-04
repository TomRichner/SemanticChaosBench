"""
Test script for basic divergence measurement
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from measurement.embeddings import get_embedding_model
from measurement.divergence import DivergenceMeasurer


def test_basic_divergence():
    """Test basic divergence measurement with simple examples"""
    print("=" * 60)
    print("Testing Basic Divergence Measurement")
    print("=" * 60)
    
    # Initialize embedding model with MPS acceleration
    print("\n1. Initializing embedding model...")
    embedding_model = get_embedding_model(model_name="all-MiniLM-L6-v2", device="auto")
    print(f"   {embedding_model}")
    
    # Initialize divergence measurer
    print("\n2. Initializing divergence measurer...")
    measurer = DivergenceMeasurer(embedding_model)
    print("   ✓ DivergenceMeasurer initialized")
    
    # Test case 1: Identical prompts with different outputs
    print("\n" + "=" * 60)
    print("Test Case 1: Identical prompts, different outputs")
    print("=" * 60)
    
    prompt1a = "What is the capital of France?"
    prompt2a = "What is the capital of France?"
    output1a = "The capital of France is Paris."
    output2a = "France's capital city is Lyon."  # Incorrect on purpose
    
    print(f"\nPrompt 1: {prompt1a}")
    print(f"Prompt 2: {prompt2a}")
    print(f"Output 1: {output1a}")
    print(f"Output 2: {output2a}")
    
    result1 = measurer.measure_single_divergence(prompt1a, prompt2a, output1a, output2a)
    
    print(f"\n📊 Results:")
    print(f"   Input distance:    {result1['input_distance']:.6f}")
    print(f"   Output distance:   {result1['output_distance']:.6f}")
    print(f"   Divergence rate:   {result1['divergence_rate']:.6f}")
    print(f"\n   ➜ Expected: Very low input distance (identical prompts)")
    print(f"   ➜ Expected: Higher output distance (different answers)")
    print(f"   ➜ Expected: Very high divergence rate (outputs diverged from identical inputs)")
    
    # Test case 2: Similar prompts with similar outputs
    print("\n" + "=" * 60)
    print("Test Case 2: Similar prompts, similar outputs")
    print("=" * 60)
    
    prompt1b = "What is the capital of France?"
    prompt2b = "Can you tell me what city is the capital of France?"
    output1b = "The capital of France is Paris."
    output2b = "Paris is the capital of France."
    
    print(f"\nPrompt 1: {prompt1b}")
    print(f"Prompt 2: {prompt2b}")
    print(f"Output 1: {output1b}")
    print(f"Output 2: {output2b}")
    
    result2 = measurer.measure_single_divergence(prompt1b, prompt2b, output1b, output2b)
    
    print(f"\n📊 Results:")
    print(f"   Input distance:    {result2['input_distance']:.6f}")
    print(f"   Output distance:   {result2['output_distance']:.6f}")
    print(f"   Divergence rate:   {result2['divergence_rate']:.6f}")
    print(f"\n   ➜ Expected: Small input distance (semantically similar prompts)")
    print(f"   ➜ Expected: Small output distance (essentially the same answer)")
    print(f"   ➜ Expected: Moderate divergence rate (stable behavior)")
    
    # Test case 3: Different prompts with very different outputs
    print("\n" + "=" * 60)
    print("Test Case 3: Different prompts, different outputs")
    print("=" * 60)
    
    prompt1c = "What is the capital of France?"
    prompt2c = "Explain quantum mechanics in simple terms."
    output1c = "The capital of France is Paris."
    output2c = "Quantum mechanics is a branch of physics that describes the behavior of matter and energy at the atomic and subatomic level."
    
    print(f"\nPrompt 1: {prompt1c}")
    print(f"Prompt 2: {prompt2c}")
    print(f"Output 1: {output1c}")
    print(f"Output 2: {output2c}")
    
    result3 = measurer.measure_single_divergence(prompt1c, prompt2c, output1c, output2c)
    
    print(f"\n📊 Results:")
    print(f"   Input distance:    {result3['input_distance']:.6f}")
    print(f"   Output distance:   {result3['output_distance']:.6f}")
    print(f"   Divergence rate:   {result3['divergence_rate']:.6f}")
    print(f"\n   ➜ Expected: Large input distance (completely different topics)")
    print(f"   ➜ Expected: Large output distance (unrelated answers)")
    print(f"   ➜ Expected: Moderate divergence rate (proportional divergence)")
    
    # Test case 4: Small perturbation (what we care about for chaos measurement)
    print("\n" + "=" * 60)
    print("Test Case 4: Tiny prompt perturbation (chaos benchmark scenario)")
    print("=" * 60)
    
    prompt1d = "Write a short story about a robot."
    prompt2d = "Write a brief story about a robot."  # "short" -> "brief"
    output1d = "In a distant future, a lonely robot named X-7 wandered through abandoned cities, searching for companionship."
    output2d = "The robot stood at the edge of the city, contemplating its existence in a world without humans."
    
    print(f"\nPrompt 1: {prompt1d}")
    print(f"Prompt 2: {prompt2d}")
    print(f"Output 1: {output1d}")
    print(f"Output 2: {output2d}")
    
    result4 = measurer.measure_single_divergence(prompt1d, prompt2d, output1d, output2d)
    
    print(f"\n📊 Results:")
    print(f"   Input distance:    {result4['input_distance']:.6f}")
    print(f"   Output distance:   {result4['output_distance']:.6f}")
    print(f"   Divergence rate:   {result4['divergence_rate']:.6f}")
    print(f"\n   ➜ Expected: Very small input distance (minimal semantic change)")
    print(f"   ➜ Expected: Variable output distance (creative task has high variance)")
    print(f"   ➜ Expected: HIGH divergence rate = CHAOTIC behavior")
    print(f"                LOW divergence rate = STABLE behavior")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of Divergence Measurements")
    print("=" * 60)
    print(f"\n{'Test Case':<40} {'Div. Rate':>12}")
    print("-" * 60)
    print(f"{'1. Identical → Different (high chaos)':<40} {result1['divergence_rate']:>12.4f}")
    print(f"{'2. Similar → Similar (stable)':<40} {result2['divergence_rate']:>12.4f}")
    print(f"{'3. Different → Different (proportional)':<40} {result3['divergence_rate']:>12.4f}")
    print(f"{'4. Tiny change → ? (THE KEY METRIC)':<40} {result4['divergence_rate']:>12.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Basic divergence measurement is working!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Integrate with model wrappers to test real LLM outputs")
    print("  - Test with multiple prompt pairs at controlled epsilon levels")
    print("  - Compare divergence rates across different models")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_divergence()

