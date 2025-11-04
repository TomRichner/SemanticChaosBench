#!/usr/bin/env python3
"""
Test script for response caching functionality

Tests cache hits/misses, cache statistics, and cache invalidation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.openai_wrapper import OpenAIModel
from src.utils.cache import Cache
import time


def print_section(title):
    """Print section divider"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_cache_basic():
    """Test basic cache operations"""
    print_section("Test 1: Basic Cache Operations")
    
    # Initialize cache
    cache = Cache(cache_dir="data/cache/test", enabled=True)
    cache.clear()  # Start fresh
    cache.reset_stats()
    
    # Test cache key generation
    key1 = cache.generate_model_cache_key(
        model_name="gpt-4o-mini",
        prompt="What is 2+2?",
        temperature=0.7,
        max_tokens=100
    )
    
    key2 = cache.generate_model_cache_key(
        model_name="gpt-4o-mini",
        prompt="What is 2+2?",
        temperature=0.7,
        max_tokens=100
    )
    
    # Same parameters should generate same key
    assert key1 == key2, "Cache keys should be identical for same parameters"
    print("✓ Cache key generation: Consistent")
    
    # Different parameters should generate different keys
    key3 = cache.generate_model_cache_key(
        model_name="gpt-4o-mini",
        prompt="What is 2+3?",  # Different prompt
        temperature=0.7,
        max_tokens=100
    )
    
    assert key1 != key3, "Cache keys should differ for different prompts"
    print("✓ Cache key generation: Differentiates prompts")
    
    # Test set and get
    test_data = {"text": "The answer is 4", "latency": 1.2}
    cache.set(key1, test_data)
    
    retrieved = cache.get(key1)
    assert retrieved == test_data, "Retrieved data should match stored data"
    print("✓ Cache set/get: Working")
    
    # Test cache miss
    nonexistent = cache.get("nonexistent_key")
    assert nonexistent is None, "Non-existent key should return None"
    print("✓ Cache miss handling: Working")
    
    # Check stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Sets: {stats['sets']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    
    # Clean up
    cache.clear()
    print("\n✓ Test 1 passed!")


def test_cache_with_model():
    """Test caching with actual model"""
    print_section("Test 2: Model Integration with Caching")
    
    try:
        # Initialize model with caching enabled
        model = OpenAIModel(model_name="gpt-4o-mini", enable_cache=True)
        
        # Clear cache stats
        model.cache.reset_stats()
        
        prompt = "Say exactly: 'Hello, testing caching!'"
        
        print("Making first API call (should be cache miss)...")
        start = time.time()
        response1 = model.generate(
            prompt=prompt,
            temperature=0.0,  # Use temperature=0 for deterministic output
            max_tokens=50
        )
        time1 = time.time() - start
        
        print(f"Response: {response1.text[:50]}...")
        print(f"Time: {time1:.3f}s")
        print(f"Latency from API: {response1.latency:.3f}s")
        
        print("\nMaking second API call with same parameters (should be cache hit)...")
        start = time.time()
        response2 = model.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=50
        )
        time2 = time.time() - start
        
        print(f"Response: {response2.text[:50]}...")
        print(f"Time: {time2:.3f}s")
        
        # Cache hit should be much faster
        assert time2 < time1 * 0.5, "Cached response should be significantly faster"
        print(f"✓ Cache speedup: {time1/time2:.1f}x faster")
        
        # Responses should be identical
        assert response1.text == response2.text, "Cached response should match original"
        print("✓ Response consistency: Identical")
        
        # Check cache stats
        stats = model.cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Sets: {stats['sets']}")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        
        assert stats['hits'] >= 1, "Should have at least one cache hit"
        assert stats['misses'] >= 1, "Should have at least one cache miss"
        
        print("\n✓ Test 2 passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires a valid OPENAI_API_KEY")
        return False
    
    return True


def test_cache_invalidation():
    """Test that cache differentiates between different parameters"""
    print_section("Test 3: Cache Invalidation")
    
    try:
        model = OpenAIModel(model_name="gpt-4o-mini", enable_cache=True)
        model.cache.reset_stats()
        
        prompt = "Count from 1 to 3"
        
        # Make three calls with different temperatures
        print("Making calls with different temperatures...")
        
        r1 = model.generate(prompt=prompt, temperature=0.0, max_tokens=50)
        print(f"Temperature 0.0: {r1.text[:30]}...")
        
        r2 = model.generate(prompt=prompt, temperature=0.5, max_tokens=50)
        print(f"Temperature 0.5: {r2.text[:30]}...")
        
        r3 = model.generate(prompt=prompt, temperature=0.0, max_tokens=50)  # Same as r1
        print(f"Temperature 0.0 (repeat): {r3.text[:30]}...")
        
        # r1 and r3 should be identical (cached)
        assert r1.text == r3.text, "Same parameters should return cached result"
        print("✓ Cache hit for identical parameters")
        
        stats = model.cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        
        # Should have at least one hit (r3 matching r1)
        assert stats['hits'] >= 1, "Should have cache hit for repeated parameters"
        
        print("\n✓ Test 3 passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires a valid OPENAI_API_KEY")
        return False
    
    return True


def test_cache_disabled():
    """Test that caching can be disabled"""
    print_section("Test 4: Cache Disabling")
    
    try:
        # Initialize model with caching disabled
        model = OpenAIModel(model_name="gpt-4o-mini", enable_cache=False)
        
        prompt = "Say: 'Testing disabled cache'"
        
        print("Making two calls with caching disabled...")
        start1 = time.time()
        r1 = model.generate(prompt=prompt, temperature=0.0, max_tokens=50)
        time1 = time.time() - start1
        
        start2 = time.time()
        r2 = model.generate(prompt=prompt, temperature=0.0, max_tokens=50)
        time2 = time.time() - start2
        
        print(f"First call: {time1:.3f}s")
        print(f"Second call: {time2:.3f}s")
        
        # Both should take similar time (no caching)
        # Allow some variation but they should be in same ballpark
        assert 0.5 < time2/time1 < 2.0, "Both calls should take similar time without caching"
        print("✓ Both calls made full API requests")
        
        # Check stats (should show no cache activity)
        stats = model.cache.get_stats()
        assert stats['enabled'] == False, "Cache should be disabled"
        print("✓ Cache properly disabled")
        
        print("\n✓ Test 4 passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires a valid OPENAI_API_KEY")
        return False
    
    return True


def main():
    """Run all cache tests"""
    print("\n" + "="*60)
    print("  CACHE TESTING SUITE")
    print("="*60)
    
    # Test 1: Basic cache operations (no API calls)
    test_cache_basic()
    
    # Tests 2-4 require API access
    print("\nThe following tests require a valid OPENAI_API_KEY.")
    print("They will make real API calls (but will be cached for cost savings).")
    
    api_tests_passed = 0
    api_tests_total = 3
    
    if test_cache_with_model():
        api_tests_passed += 1
    
    if test_cache_invalidation():
        api_tests_passed += 1
    
    if test_cache_disabled():
        api_tests_passed += 1
    
    # Summary
    print_section("Test Summary")
    print(f"Basic tests: ✓ Passed")
    print(f"API tests: {api_tests_passed}/{api_tests_total} passed")
    
    if api_tests_passed == api_tests_total:
        print("\n✓ All tests passed!")
        return 0
    elif api_tests_passed > 0:
        print(f"\n⚠ Some API tests failed or skipped")
        return 1
    else:
        print("\n✗ API tests could not run (check API key)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

