#!/usr/bin/env python
"""
This script prepares the environment for running BERTopic tests in CI.
It ensures all the required dependencies are installed and available.
"""

import importlib.util
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependency(module_name):
    """Check if a module is available."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def main():
    """Prepare the environment for BERTopic tests."""
    dependencies = {
        "bertopic": "bertopic>=0.15.0",
        "umap-learn": "umap-learn>=0.5.3",
        "hdbscan": "hdbscan>=0.8.29",
        "sentence-transformers": "sentence-transformers>=2.2.2",
        "scikit-learn": "scikit-learn>=1.0.0",
        "numpy": "numpy>=1.20.0",
        "scipy": "scipy>=1.7.0",
        "plotly": "plotly>=5.0.0",
    }
    
    missing = []
    for module, package in dependencies.items():
        if not check_dependency(module):
            missing.append(package)
    
    if missing:
        logger.info(f"Missing dependencies: {', '.join(missing)}")
        for package in missing:
            if not install_package(package):
                logger.error(f"Failed to install {package}")
                return 1
        
        # Verify installation
        for module in dependencies.keys():
            if not check_dependency(module):
                logger.error(f"Failed to install {module}")
                return 1
    
    # All dependencies are installed
    logger.info("All dependencies are installed")
    
    # Try to import BERTopic components to verify
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from bertopic.representation import KeyBERTInspired
        
        logger.info("BERTopic imports successful!")
    except ImportError as e:
        logger.error(f"Error importing BERTopic components: {e}")
        return 1
    
    # Try to create a BERTopic model
    try:
        model = BERTopic(nr_topics=3)
        logger.info("BERTopic model created successfully!")
    except Exception as e:
        logger.error(f"Error creating BERTopic model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())