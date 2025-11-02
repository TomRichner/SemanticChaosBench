#!/usr/bin/env python
"""
Pilot study: Initial validation experiment with 10 prompt pairs on 2 models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("pilot_study")


def main():
    """Run pilot study"""
    logger.info("Starting pilot study...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config.keys()}")
    
    # TODO: Implement pilot study
    logger.warning("Pilot study not yet implemented")
    logger.info("Please complete Phase 1 tasks first:")
    logger.info("  - Configure Sentence-BERT with MPS acceleration")
    logger.info("  - Implement prompt perturbation generator")
    logger.info("  - Create unified model API interface")
    logger.info("  - Build basic divergence measurement")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

