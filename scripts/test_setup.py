#!/usr/bin/env python
"""
Test script to verify environment setup
"""

import sys
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


def test_pytorch():
    """Test PyTorch installation and MPS availability"""
    print("=" * 60)
    print("PyTorch Configuration")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) is available!")
        device = "mps"
    else:
        print("⚠ MPS not available, will use CPU")
        device = "cpu"
    
    return device


def test_sentence_transformers(device):
    """Test Sentence-BERT model loading"""
    print("\n" + "=" * 60)
    print("Sentence-BERT Configuration")
    print("=" * 60)
    
    try:
        print("Loading model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"✓ Model loaded successfully on device: {device}")
        
        # Test encoding
        test_sentences = ["This is a test sentence.", "This is another test."]
        embeddings = model.encode(test_sentences)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def test_api_keys():
    """Test API key configuration"""
    print("\n" + "=" * 60)
    print("API Keys Configuration")
    print("=" * 60)
    
    load_dotenv()
    
    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google AI Studio": "GOOGLE_API_KEY",
        "Replicate": "REPLICATE_API_TOKEN",
        "Together": "TOGETHER_API_KEY",
    }
    
    for name, key in api_keys.items():
        value = os.getenv(key)
        if value and value != "sk-...":
            print(f"✓ {name}: Configured")
        else:
            print(f"⚠ {name}: Not configured (set {key} in .env)")


def test_project_structure():
    """Test project structure"""
    print("\n" + "=" * 60)
    print("Project Structure")
    print("=" * 60)
    
    required_dirs = [
        "src",
        "src/perturbation",
        "src/models",
        "src/measurement",
        "src/analysis",
        "src/utils",
        "scripts",
        "experiments/configs",
        "experiments/results",
        "data/prompt_pairs",
        "data/outputs",
        "data/cache",
        "notebooks",
        "tests",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all setup tests"""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Semantic Chaos Bench - Setup Test" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    
    device = test_pytorch()
    sbert_ok = test_sentence_transformers(device)
    test_api_keys()
    structure_ok = test_project_structure()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if sbert_ok and structure_ok:
        print("✓ All critical tests passed!")
        print("\nNext steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: python scripts/pilot_study.py")
        return 0
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

