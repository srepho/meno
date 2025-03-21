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

def install_package(package_name, extra_args=None):
    """Install a package using pip with optional extra arguments."""
    logger.info(f"Installing {package_name}...")
    cmd = [sys.executable, "-m", "pip", "install"]
    
    # Add extra args if provided
    if extra_args:
        cmd.extend(extra_args)
    
    cmd.append(package_name)
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def main():
    """Prepare the environment for BERTopic tests."""
    # Install base dependencies first
    base_dependencies = [
        "numpy>=1.20.0",
        "scipy>=1.7.0,<1.9.0",  # Specific version range to avoid triu import error
        "scikit-learn>=1.0.0",
        "pynndescent>=0.5.7",
        "numba>=0.55.1",
        "llvmlite>=0.38.0"
    ]
    
    # Install these first to avoid dependency conflicts
    logger.info("Installing base dependencies first...")
    for package in base_dependencies:
        if not install_package(package, ["--no-cache-dir"]):
            logger.error(f"Failed to install base dependency {package}")
            return 1
    
    # Define dependency mapping for verification
    dependency_mapping = {
        "umap-learn": "umap",
        "hdbscan": "hdbscan",
        "sentence-transformers": "sentence_transformers",
        "plotly": "plotly",
        "bertopic": "bertopic",
    }
    
    # Now install the core dependencies in the correct order
    ordered_dependencies = [
        # First install UMAP with specific flags
        ("umap-learn", ["--no-cache-dir"]),
        # Then install HDBSCAN
        ("hdbscan>=0.8.29", ["--no-cache-dir"]),
        # Install sentence-transformers
        ("sentence-transformers>=2.2.2", None),
        # Install plotly
        ("plotly>=5.0.0", None),
        # Finally install BERTopic
        ("bertopic>=0.15.0", None)
    ]
    
    logger.info("Installing main dependencies in order...")
    for package, extra_args in ordered_dependencies:
        pkg_name = package.split(">=")[0]
        module_name = dependency_mapping.get(pkg_name, pkg_name)
        
        if not check_dependency(module_name):
            logger.info(f"Installing {package}...")
            if not install_package(package, extra_args):
                logger.error(f"Failed to install {package}")
                # Continue anyway to see if we can install the others
                continue
    
    # Verify all core dependencies are installed
    missing_deps = []
    for pkg_name, module_name in dependency_mapping.items():
        if not check_dependency(module_name):
            missing_deps.append(pkg_name)
    
    if missing_deps:
        logger.error(f"Failed to install these dependencies: {', '.join(missing_deps)}")
        logger.info("Continuing anyway to see if tests can run with partial dependencies")
    else:
        logger.info("All dependencies are installed successfully")
    
    # Try to import BERTopic components to verify
    bertopic_success = False
    try:
        if check_dependency("bertopic"):
            logger.info("Trying to import BERTopic components...")
            from bertopic import BERTopic
            bertopic_success = True
            
            try:
                from bertopic.vectorizers import ClassTfidfTransformer
                from bertopic.representation import KeyBERTInspired
                logger.info("BERTopic imports successful!")
                
                # Try to create a BERTopic model
                try:
                    model = BERTopic(nr_topics=3)
                    logger.info("BERTopic model created successfully!")
                except Exception as e:
                    logger.error(f"Error creating BERTopic model: {e}")
                    # Continue anyway
            except ImportError as e:
                logger.error(f"Error importing BERTopic components: {e}")
                # Continue anyway
    except ImportError as e:
        logger.error(f"Error importing BERTopic: {e}")
        # Continue anyway
    
    # Try to import UMAP to verify
    umap_success = False
    try:
        if check_dependency("umap"):
            logger.info("Trying to import UMAP...")
            import umap
            logger.info("UMAP import successful!")
            umap_success = True
    except ImportError as e:
        logger.error(f"Error importing UMAP: {e}")
    
    # Try to import HDBSCAN to verify
    hdbscan_success = False
    try:
        if check_dependency("hdbscan"):
            logger.info("Trying to import HDBSCAN...")
            import hdbscan
            logger.info("HDBSCAN import successful!")
            hdbscan_success = True
    except ImportError as e:
        logger.error(f"Error importing HDBSCAN: {e}")
    
    # Summary
    logger.info(f"UMAP available: {umap_success}")
    logger.info(f"HDBSCAN available: {hdbscan_success}")
    logger.info(f"BERTopic available: {bertopic_success}")
    
    # If we have UMAP and HDBSCAN, that's good enough for many tests
    if umap_success and hdbscan_success:
        logger.info("Core dependencies UMAP and HDBSCAN are available - tests can run")
        return 0
    else:
        logger.warning("Some core dependencies are missing - some tests may fail")
        return 0  # Continue anyway to let tests be skipped properly

if __name__ == "__main__":
    sys.exit(main())