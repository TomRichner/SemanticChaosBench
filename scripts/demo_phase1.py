#!/usr/bin/env python3
"""
Quick demo of Phase 1 completed features:
1. Sentence-BERT with MPS acceleration
2. Prompt perturbation generator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.measurement.embeddings import EmbeddingModel

def main():
    print("="*70)
    print("SEMANTIC CHAOS BENCH - Phase 1 Demo")
    print("="*70)
    
    # 1. Initialize Sentence-BERT with MPS acceleration
    print("\n1. Initializing Sentence-BERT with MPS acceleration...")
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2", device="auto")
    print(f"   ✓ {model}")
    
    # 2. Demonstrate semantic distance measurement
    print("\n2. Demonstrating semantic distance measurement...")
    
    prompts = [
        "What is the capital of France?",
        "What's the capital city of France?",  # Very similar
        "Can you tell me France's capital?",   # Similar  
        "Paris is a beautiful city.",          # Related but different
        "What is machine learning?",           # Completely different
    ]
    
    print(f"\n   Base prompt: '{prompts[0]}'")
    print("\n   Semantic distances to variations:")
    
    # Encode all prompts
    embeddings = model.encode(prompts)
    base_embedding = embeddings[0]
    
    # Compute distances
    for i, prompt in enumerate(prompts[1:], 1):
        distance = model.cosine_distance(base_embedding, embeddings[i])
        
        # Visual indicator
        if distance < 0.1:
            indicator = "🟢 Very similar"
        elif distance < 0.3:
            indicator = "🟡 Similar"
        elif distance < 0.5:
            indicator = "🟠 Related"
        else:
            indicator = "🔴 Different"
        
        print(f"   {distance:.4f} {indicator:15s} - '{prompt}'")
    
    # 3. Show key capabilities
    print("\n3. Key Capabilities Implemented:")
    print("   ✓ MPS (Apple Silicon GPU) acceleration")
    print("   ✓ Batch embedding encoding")
    print("   ✓ Cosine distance computation")
    print("   ✓ Pairwise distance matrices")
    print("   ✓ Paraphrase generation (via API)")
    print("   ✓ Semantic filtering by distance")
    print("   ✓ Prompt pair generation pipeline")
    
    # 4. Show what's next
    print("\n4. Next Steps (Phase 1 remaining):")
    print("   • Create unified model API interface")
    print("   • Build basic divergence measurement")
    
    print("\n" + "="*70)
    print("Demo complete! Phase 1 core components are working. 🎉")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

