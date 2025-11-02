"""
Caching utilities for API responses
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional


class Cache:
    """Simple file-based cache for API responses"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(value, f, indent=2)

