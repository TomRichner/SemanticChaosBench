"""
Test script for unified model interface

This script demonstrates and validates the unified model API interface.
It tests the model router, factory function, and individual model wrappers.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.models import ModelInterface, get_model, ModelResponse

# Load environment variables
load_dotenv()


def test_model_routing():
    """Test that model names are correctly routed to appropriate wrappers"""
    print("=" * 70)
    print("Testing Model Routing")
    print("=" * 70)
    
    # Tom selected these models.  Do not change them.  These are the correct model strings
    test_cases = [
        ("gpt-4o-mini", "OpenAIModel"),
        ("claude-haiku-4-5", "AnthropicModel"),
        ("gemini-2.5-flash", "GoogleModel"),
        ("meta/meta-llama-3-8b-instruct", "ReplicateModel"),  # Updated: correct Replicate format
        ("meta-llama/Meta-Llama-3-8B-Instruct-Lite", "TogetherModel"),
    ]
    
    for model_name, expected_class in test_cases:
        try:
            model = get_model(model_name)
            actual_class = model.__class__.__name__
            status = "✓" if actual_class == expected_class else "✗"
            print(f"{status} {model_name:45} -> {actual_class}")
            
            if actual_class != expected_class:
                print(f"  ERROR: Expected {expected_class}, got {actual_class}")
        except ValueError as e:
            if "API key required" in str(e):
                print(f"⚠ {model_name:45} -> Skipped (no API key)")
            else:
                print(f"✗ {model_name:45} -> Error: {e}")
        except Exception as e:
            print(f"✗ {model_name:45} -> Error: {e}")
    
    print()


def test_google_generation():
    """Test Google model generation (if API key available)"""
    print("=" * 70)
    print("Testing Google Gemini Generation")
    print("=" * 70)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠ GOOGLE_API_KEY not set - skipping test\n")
        return
    
    try:
        # Test using factory function
        model = get_model("gemini-2.5-flash")
        print(f"✓ Created model: {model.__class__.__name__}")
        
        # Test generation
        prompt = "In one sentence, what is the capital of France?"
        print(f"\nPrompt: {prompt}")
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out after 30 seconds")
        
        # Set 30-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            response = model.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=500
            )
            signal.alarm(0)  # Cancel alarm
            
            print(f"\nResponse:")
            print(f"  Text: {response.text}")
            print(f"  Latency: {response.latency:.2f}s")
            print(f"  Tokens: {response.token_count}")
            print(f"  Model: {response.model_name}")
            print()
        except TimeoutError as e:
            signal.alarm(0)
            print(f"✗ Timeout: {e}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")


def test_unified_interface():
    """Test the unified ModelInterface class"""
    print("=" * 70)
    print("Testing Unified ModelInterface")
    print("=" * 70)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠ GOOGLE_API_KEY not set - skipping test\n")
        return
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out after 30 seconds")
        
        # Create unified interface
        interface = ModelInterface()
        print("✓ Created ModelInterface instance")
        
        # Test generation
        prompt = "In one sentence, what is 2+2?"
        print(f"\nPrompt: {prompt}")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            response = interface.generate(
                prompt=prompt,
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=500
            )
            signal.alarm(0)
            
            print(f"\nResponse:")
            print(f"  Text: {response.text}")
            print(f"  Latency: {response.latency:.2f}s")
            print(f"  Tokens: {response.token_count}")
            print()
        except TimeoutError as e:
            signal.alarm(0)
            print(f"✗ Timeout: {e}\n")
            return
        
        # Test with system prompt
        signal.alarm(30)
        try:
            response2 = interface.generate(
                prompt="What is quantum computing?",
                model="gemini-2.5-flash",
                temperature=0.5,
                max_tokens=500,
                system_prompt="You are a helpful assistant. Be concise."
            )
            signal.alarm(0)
            
            print("With system prompt:")
            print(f"  Text: {response2.text[:100]}...")
            print(f"  Latency: {response2.latency:.2f}s")
            print()
        except TimeoutError as e:
            signal.alarm(0)
            print(f"✗ Timeout: {e}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")


def test_multiple_models():
    """Test generation across multiple models (if API keys available)"""
    print("=" * 70)
    print("Testing Multiple Models")
    print("=" * 70)
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Generation timed out")
    
    interface = ModelInterface()
    prompt = "In one sentence, what is the speed of light?"
    
    models_to_test = [
        ("gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("gpt-4o-mini", "OPENAI_API_KEY"),
        ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        ("meta/meta-llama-3-8b-instruct", "REPLICATE_API_TOKEN"),  # Updated: correct Replicate format
        ("meta-llama/Meta-Llama-3-8B-Instruct-Lite", "TOGETHER_API_KEY"),
    ]
    
    for model_name, env_key in models_to_test:
        if not os.getenv(env_key):
            print(f"⚠ {model_name:35} - Skipped (no {env_key})")
            continue
        
        try:
            # Set 30-second timeout for each model
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            response = interface.generate(
                prompt=prompt,
                model=model_name,
                temperature=0.0,
                max_tokens=1000
            )
            
            signal.alarm(0)  # Cancel alarm
            print(f"✓ {model_name:35} - {response.latency:.2f}s - {len(response.text)} chars")
            
        except TimeoutError:
            signal.alarm(0)
            print(f"✗ {model_name:35} - Timeout after 30s")
        except Exception as e:
            signal.alarm(0)
            print(f"✗ {model_name:35} - Error: {e}")
    
    print()


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("=" * 70)
    print("Testing Error Handling")
    print("=" * 70)
    
    # Test invalid model name
    try:
        model = get_model("invalid-model-xyz")
        print("✗ Should have raised ValueError for invalid model")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid model")
        print(f"  Message: {str(e)[:60]}...")
    
    # Test missing API key
    try:
        # Temporarily remove API key
        original_key = os.environ.pop("GOOGLE_API_KEY", None)
        model = get_model("gemini-2.5-flash")
        print("✗ Should have raised ValueError for missing API key")
        # Restore key
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for missing API key")
        print(f"  Message: {str(e)[:60]}...")
        # Restore key
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key
    
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("UNIFIED MODEL INTERFACE TEST SUITE")
    print("=" * 70)
    print()
    
    # Check for API keys
    api_keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN"),
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
    }
    
    print("API Key Status:")
    for key, value in api_keys.items():
        status = "✓ Set" if value else "✗ Not set"
        print(f"  {status:12} {key}")
    print()
    
    # Run tests
    test_model_routing()
    test_google_generation()
    test_unified_interface()
    test_multiple_models()
    test_error_handling()
    
    print("=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

