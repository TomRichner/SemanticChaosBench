"""
API helper utilities for rate limiting, retries, and error handling
"""

import time
import threading
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    exponential_backoff: bool = True
    max_delay: float = 60.0


class RateLimiter:
    """
    Thread-safe rate limiter for API calls
    
    Enforces minimum delay between requests per provider
    """
    
    def __init__(self, min_delay: float = 0.5):
        """
        Initialize rate limiter
        
        Args:
            min_delay: Minimum seconds between requests
        """
        self.min_delay = min_delay
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to enforce rate limit
        
        Returns:
            Time waited in seconds
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                time.sleep(wait_time)
                self.last_request_time = time.time()
                return wait_time
            else:
                self.last_request_time = current_time
                return 0.0


# Global rate limiters for each provider
_rate_limiters = {}
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(provider: str, min_delay: float = 0.5) -> RateLimiter:
    """
    Get or create rate limiter for a provider
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        min_delay: Minimum delay between requests
        
    Returns:
        RateLimiter instance for the provider
    """
    with _rate_limiter_lock:
        if provider not in _rate_limiters:
            _rate_limiters[provider] = RateLimiter(min_delay)
        return _rate_limiters[provider]


def is_retryable_error(exception: Exception, retryable_exceptions: Tuple[Type[Exception], ...]) -> bool:
    """
    Check if an exception is retryable
    
    Args:
        exception: Exception to check
        retryable_exceptions: Tuple of exception types that are retryable
        
    Returns:
        True if exception should trigger retry
    """
    # Check if exception is an instance of any retryable exception type
    if isinstance(exception, retryable_exceptions):
        return True
    
    # Check for common retryable HTTP status codes in error messages
    error_msg = str(exception).lower()
    retryable_codes = ['500', '502', '503', '504', '429']
    
    for code in retryable_codes:
        if code in error_msg:
            return True
    
    # Check for common transient error keywords
    transient_keywords = [
        'timeout',
        'connection reset',
        'connection refused',
        'temporary failure',
        'service unavailable',
        'internal server error',
        'rate limit',
        'too many requests'
    ]
    
    for keyword in transient_keywords:
        if keyword in error_msg:
            return True
    
    return False


def retry_with_backoff(
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    config: Optional[RetryConfig] = None
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff
    
    Args:
        retryable_exceptions: Tuple of exception types to retry on
        config: Retry configuration (uses defaults if None)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(
            retryable_exceptions=(ConnectionError, TimeoutError),
            config=RetryConfig(max_retries=3, base_delay=1.0)
        )
        def make_api_call():
            # ... API call code
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if not is_retryable_error(e, retryable_exceptions):
                        # Non-retryable error, raise immediately
                        raise
                    
                    # Last attempt, don't sleep
                    if attempt == config.max_retries - 1:
                        break
                    
                    # Calculate delay
                    if config.exponential_backoff:
                        delay = min(
                            config.base_delay * (2 ** attempt),
                            config.max_delay
                        )
                    else:
                        delay = config.base_delay
                    
                    # Log retry attempt (optional - can be enhanced with proper logging)
                    print(f"Attempt {attempt + 1}/{config.max_retries} failed: {str(e)}")
                    print(f"Retrying in {delay:.1f}s...")
                    
                    time.sleep(delay)
            
            # All retries exhausted
            if last_exception:
                raise RuntimeError(
                    f"Failed after {config.max_retries} attempts. "
                    f"Last error: {str(last_exception)}"
                ) from last_exception
            
            # This shouldn't happen, but just in case
            raise RuntimeError("Retry logic failed unexpectedly")
        
        return wrapper
    return decorator


def classify_api_error(exception: Exception) -> dict:
    """
    Classify an API error and provide helpful information
    
    Args:
        exception: Exception to classify
        
    Returns:
        Dictionary with error classification information
    """
    error_msg = str(exception).lower()
    
    classification = {
        'is_retryable': False,
        'category': 'unknown',
        'suggestion': 'Please check the error message and try again.',
        'original_error': str(exception)
    }
    
    # Authentication errors
    if any(kw in error_msg for kw in ['401', 'unauthorized', 'authentication', 'invalid api key']):
        classification.update({
            'category': 'authentication',
            'suggestion': 'Check your API key in .env file.'
        })
    
    # Permission/quota errors
    elif any(kw in error_msg for kw in ['403', 'forbidden', 'quota', 'billing']):
        classification.update({
            'category': 'permission',
            'suggestion': 'Check your account permissions, quota, and billing status.'
        })
    
    # Rate limiting
    elif any(kw in error_msg for kw in ['429', 'rate limit', 'too many requests']):
        classification.update({
            'is_retryable': True,
            'category': 'rate_limit',
            'suggestion': 'Rate limit exceeded. Consider increasing delays between requests.'
        })
    
    # Server errors
    elif any(kw in error_msg for kw in ['500', '502', '503', '504', 'internal server error', 'service unavailable']):
        classification.update({
            'is_retryable': True,
            'category': 'server_error',
            'suggestion': 'Server error occurred. Will retry automatically.'
        })
    
    # Timeout errors
    elif any(kw in error_msg for kw in ['timeout', 'timed out']):
        classification.update({
            'is_retryable': True,
            'category': 'timeout',
            'suggestion': 'Request timed out. Consider increasing timeout value.'
        })
    
    # Invalid request
    elif any(kw in error_msg for kw in ['400', 'bad request', 'invalid']):
        classification.update({
            'category': 'invalid_request',
            'suggestion': 'Check your request parameters (prompt, temperature, max_tokens, etc.).'
        })
    
    # Connection errors
    elif any(kw in error_msg for kw in ['connection', 'network']):
        classification.update({
            'is_retryable': True,
            'category': 'connection',
            'suggestion': 'Network connection issue. Check your internet connection.'
        })
    
    return classification

