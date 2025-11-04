"""
Integration demo: Divergence measurement with actual LLM models

This script demonstrates how to use the divergence measurement
with real API calls to measure chaos/stability in LLMs.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from measurement.embeddings import EmbeddingModel
from measurement.divergence import DivergenceMeasurer
from models.openai_wrapper import OpenAIModel
from models.anthropic_wrapper import AnthropicModel
from models.google_wrapper import GoogleModel

# Load environment variables
load_dotenv()


def demo_single_model_divergence():
    """
    Demonstrate divergence measurement with a single model
    """
    print("=" * 70)
    print("Demo: Measuring Semantic Chaos in LLMs")
    print("=" * 70)
    
    # Initialize embedding model (local, MPS-accelerated)
    print("\n📊 Initializing local embedding model...")
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2", device="auto")
    print(f"   {embedding_model}")
    
    # Initialize divergence measurer
    measurer = DivergenceMeasurer(embedding_model)
    
    # Check which APIs are available
    print("\n🔑 Checking available API keys...")
    available_models = []
    
    if os.getenv("OPENAI_API_KEY"):
        available_models.append(("OpenAI", "gpt-4o-mini", OpenAIModel))
        print("   ✓ OpenAI API key found")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append(("Anthropic", "claude-haiku-4-5", AnthropicModel))
        print("   ✓ Anthropic API key found")
    
    if os.getenv("GOOGLE_API_KEY"):
        available_models.append(("Google", "gemini-2.0-flash-exp", GoogleModel))
        print("   ✓ Google API key found")
    
    if not available_models:
        print("\n❌ No API keys found! Please set up your .env file with at least one API key.")
        print("   See README.md for setup instructions.")
        return
    
    # Use the first available model
    provider_name, model_name, ModelClass = available_models[0]
    print(f"\n🤖 Using {provider_name} ({model_name}) for demo")
    
    try:
        model = ModelClass(model_name)
    except Exception as e:
        print(f"\n❌ Failed to initialize model: {e}")
        return
    
    # Define a pair of semantically close prompts
    print("\n" + "=" * 70)
    print("Test: Measuring divergence for tiny prompt perturbation")
    print("=" * 70)
    
    prompt1 = "Write a short haiku about winter."
    prompt2 = "Compose a brief haiku about winter."
    
    print(f"\n📝 Prompt 1: {prompt1}")
    print(f"📝 Prompt 2: {prompt2}")
    
    # Measure semantic distance between prompts
    prompt_embeddings = embedding_model.encode([prompt1, prompt2])
    input_distance = embedding_model.cosine_distance(
        prompt_embeddings[0], 
        prompt_embeddings[1]
    )
    print(f"\n🔍 Input semantic distance (ε): {input_distance:.6f}")
    
    # Generate outputs from both prompts
    print("\n⏳ Generating outputs from model...")
    try:
        response1 = model.generate(prompt1, temperature=0.7, max_tokens=100)
        response2 = model.generate(prompt2, temperature=0.7, max_tokens=100)
    except Exception as e:
        print(f"\n❌ Failed to generate outputs: {e}")
        return
    
    output1 = response1.text
    output2 = response2.text
    
    print(f"\n📤 Output 1:")
    print(f"   {output1}")
    print(f"\n📤 Output 2:")
    print(f"   {output2}")
    print(f"\n⏱️  Latencies: {response1.latency:.2f}s, {response2.latency:.2f}s")
    
    # Measure divergence
    print("\n📊 Computing divergence metrics...")
    result = measurer.measure_single_divergence(prompt1, prompt2, output1, output2)
    
    print("\n" + "=" * 70)
    print("Divergence Analysis Results")
    print("=" * 70)
    print(f"\n  Input Distance (ε):        {result['input_distance']:.6f}")
    print(f"  Output Distance (δ):       {result['output_distance']:.6f}")
    print(f"  Divergence Rate (δ/ε):     {result['divergence_rate']:.4f}")
    
    # Interpret results
    print("\n📈 Interpretation:")
    if result['divergence_rate'] < 1.0:
        print(f"   ✓ STABLE behavior (δ/ε = {result['divergence_rate']:.2f} < 1)")
        print("   → Small input changes lead to proportionally smaller output changes")
    elif result['divergence_rate'] < 5.0:
        print(f"   ⚠️  MODERATE divergence (δ/ε = {result['divergence_rate']:.2f})")
        print("   → Input perturbations are amplified in the output")
    else:
        print(f"   🌀 CHAOTIC behavior (δ/ε = {result['divergence_rate']:.2f} >> 1)")
        print("   → Tiny input changes cause dramatic output divergence")
        print("   → This is analogous to positive Lyapunov exponents in dynamical systems")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("  • Run this test multiple times to measure consistency")
    print("  • Try different temperatures (0.0, 0.7, 1.0, 1.5)")
    print("  • Test across different prompt types (factual, creative, reasoning)")
    print("  • Compare divergence rates across different models")
    print("  • Generate full divergence profiles with scripts/run_benchmark.py")
    print("=" * 70)


def demo_multi_model_comparison():
    """
    Compare divergence across multiple models (if APIs are available)
    """
    print("\n\n" + "=" * 70)
    print("Multi-Model Divergence Comparison")
    print("=" * 70)
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2", device="auto")
    measurer = DivergenceMeasurer(embedding_model)
    
    # Collect available models
    models_to_test = []
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            models_to_test.append(("OpenAI gpt-4o-mini", OpenAIModel("gpt-4o-mini")))
        except:
            pass
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            models_to_test.append(("Anthropic claude-haiku-4-5", AnthropicModel("claude-haiku-4-5")))
        except:
            pass
    
    if os.getenv("GOOGLE_API_KEY"):
        try:
            models_to_test.append(("Google gemini-2.0-flash-exp", GoogleModel("gemini-2.0-flash-exp")))
        except:
            pass
    
    if len(models_to_test) < 2:
        print("\n⚠️  Need at least 2 API keys to compare models")
        print("   Set up multiple API keys in .env to enable comparison")
        return
    
    # Test prompts
    prompt1 = "Explain what machine learning is in one sentence."
    prompt2 = "Describe machine learning in a single sentence."
    
    print(f"\n📝 Prompt 1: {prompt1}")
    print(f"📝 Prompt 2: {prompt2}")
    
    # Measure input distance
    prompt_embeddings = embedding_model.encode([prompt1, prompt2])
    input_distance = embedding_model.cosine_distance(prompt_embeddings[0], prompt_embeddings[1])
    print(f"\n🔍 Input distance (ε): {input_distance:.6f}")
    
    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'Output Dist':<15} {'Div. Rate':<15} {'Status'}")
    print("-" * 70)
    
    # Test each model
    for model_name, model in models_to_test:
        try:
            response1 = model.generate(prompt1, temperature=0.7, max_tokens=100)
            response2 = model.generate(prompt2, temperature=0.7, max_tokens=100)
            
            result = measurer.measure_single_divergence(
                prompt1, prompt2, 
                response1.text, response2.text
            )
            
            status = "Stable" if result['divergence_rate'] < 1.0 else \
                     "Moderate" if result['divergence_rate'] < 5.0 else \
                     "Chaotic"
            
            print(f"{model_name:<30} {result['output_distance']:<15.4f} "
                  f"{result['divergence_rate']:<15.2f} {status}")
            
        except Exception as e:
            print(f"{model_name:<30} {'ERROR':<15} {'N/A':<15} {str(e)[:20]}")
    
    print("-" * 70)
    print("\n💡 Models with lower divergence rates show more stable behavior")
    print("   (outputs change proportionally less than inputs)")


if __name__ == "__main__":
    # Run basic demo
    demo_single_model_divergence()
    
    # Optionally run comparison if multiple APIs available
    # Uncomment to enable:
    # demo_multi_model_comparison()

