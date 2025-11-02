#!/usr/bin/env python
"""
Quick test of Google AI Studio API integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.google_wrapper import GoogleModel
from dotenv import load_dotenv


def main():
    """Test Google AI Studio wrapper"""
    print("=" * 60)
    print("Testing Google AI Studio (Gemini) Integration")
    print("=" * 60)
    
    load_dotenv()
    
    try:
        # Initialize model
        print("\n1. Initializing GoogleModel with gemini-2.5-flash...")
        model = GoogleModel(model_name="gemini-2.5-flash")
        print("✓ Model initialized successfully")
        
        # Test generation
        print("\n2. Testing text generation...")
        test_prompt = "Say hello and confirm you are Gemini in one sentence."
        
        response = model.generate(
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        print("✓ Generation successful!")
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response.text}")
        print(f"Latency: {response.latency:.3f}s")
        print(f"Tokens: {response.token_count}")
        print(f"Model: {response.model_name}")
        
        print("\n" + "=" * 60)
        print("✓ Google AI Studio integration working correctly!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify GOOGLE_API_KEY is set in .env")
        print("2. Get your API key from: https://ai.google.dev")
        print("3. Check API key has proper permissions")
        return 1


if __name__ == "__main__":
    sys.exit(main())

