#!/usr/bin/env python
"""
Full benchmark runner
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger

logger = setup_logger("run_benchmark")


def main():
    """Run full benchmark suite"""
    logger.info("Starting full benchmark...")
    
    # TODO: Implement full benchmark
    logger.warning("Full benchmark not yet implemented")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

