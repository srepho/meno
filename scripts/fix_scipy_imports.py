#!/usr/bin/env python
"""
This script patches scipy import issues related to the triu function.
It creates a compatibility layer for older code expecting triu in scipy.linalg.special_matrices.
"""

import importlib.util
import logging
import sys

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_patch_scipy():
    """
    Check if scipy.linalg.special_matrices exists and if it has triu.
    If not, create a patch that imports triu from scipy.linalg.
    """
    logger.info("Checking scipy version and structure...")
    
    try:
        import scipy
        logger.info(f"Scipy version: {scipy.__version__}")
        
        # First check if triu is in special_matrices
        try:
            from scipy.linalg.special_matrices import triu
            logger.info("triu found in scipy.linalg.special_matrices - no patch needed")
            return True
        except ImportError:
            logger.info("Could not import triu from scipy.linalg.special_matrices")
            
            # Check if triu is in scipy.linalg
            try:
                from scipy.linalg import triu
                logger.info("triu found in scipy.linalg - can create compatibility patch")
                
                # Create patch module
                try:
                    import types
                    import sys
                    import scipy.linalg.special_matrices
                    
                    # Check if module exists but just doesn't have triu
                    if 'triu' not in dir(scipy.linalg.special_matrices):
                        logger.info("Creating compatibility patch for scipy.linalg.special_matrices.triu")
                        setattr(scipy.linalg.special_matrices, 'triu', triu)
                        logger.info("Patch applied successfully!")
                        return True
                except Exception as e:
                    logger.error(f"Error creating patch: {e}")
                    return False
            except ImportError:
                logger.error("triu not found in scipy.linalg either")
                return False
    except ImportError:
        logger.error("Could not import scipy")
        return False

if __name__ == "__main__":
    if check_and_patch_scipy():
        logger.info("Scipy patching completed successfully")
        sys.exit(0)
    else:
        logger.error("Failed to patch scipy imports")
        sys.exit(1)