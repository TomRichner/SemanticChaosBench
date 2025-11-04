"""
Caching utilities for API responses
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import asdict


class CacheStats:
    """Track cache statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"sets={self.sets}, hit_rate={self.hit_rate:.2%})")


class Cache:
    """File-based cache for API responses with statistics tracking"""
    
    def __init__(self, cache_dir: str = "data/cache", enabled: bool = True):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled (can disable for testing)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.stats = CacheStats()
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MD5 hash of serialized arguments
        """
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def generate_model_cache_key(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Generate cache key for model generation request
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Cache key string
        """
        # Include all parameters that affect the output
        cache_params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs  # Include any additional parameters
        }
        return self._get_cache_key(**cache_params)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            self.stats.misses += 1
            return None
        
        try:
            with open(cache_file, 'r') as f:
                self.stats.hits += 1
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If cache file is corrupted, treat as miss
            print(f"Warning: Failed to read cache file {key}: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
        """
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(value, f, indent=2)
            self.stats.sets += 1
        except (TypeError, IOError) as e:
            print(f"Warning: Failed to write cache file {key}: {e}")
    
    def clear(self) -> int:
        """
        Clear all cache files
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "hit_rate": self.stats.hit_rate,
            "enabled": self.enabled
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self.stats = CacheStats()

