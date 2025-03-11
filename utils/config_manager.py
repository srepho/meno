"""Configuration management for Meno.

This module provides functionality for loading, saving, and managing
configuration settings, including first-run detection and automatic
configuration based on system capabilities.
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import shutil

logger = logging.getLogger(__name__)


def get_user_config_dir() -> Path:
    """Get or create user configuration directory.
    
    The user configuration directory is located at ~/.meno and contains
    user-specific settings, cached models, and other data.
    
    Returns
    -------
    Path
        Path to the user configuration directory
    """
    user_dir = Path.home() / ".meno"
    user_dir.mkdir(exist_ok=True)
    
    # Create subdirectories if they don't exist
    for subdir in ["config", "cache", "models"]:
        (user_dir / subdir).mkdir(exist_ok=True)
    
    return user_dir


def is_first_run() -> bool:
    """Check if this is the first run of Meno.
    
    Returns
    -------
    bool
        True if this is the first run, False otherwise
    """
    config_marker = get_user_config_dir() / "initialized"
    return not config_marker.exists()


def mark_initialized() -> None:
    """Mark Meno as initialized."""
    config_marker = get_user_config_dir() / "initialized"
    config_marker.touch()


def analyze_system_capabilities() -> Dict[str, Any]:
    """Analyze system capabilities to optimize default settings.
    
    This function checks system resources like memory and GPU availability
    to set appropriate default configuration values.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of configuration settings based on system capabilities
    """
    config = {}
    
    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        
        # Adjust batch sizes based on available memory
        if total_gb < 4:  # Low memory
            config["modeling"] = {
                "embeddings": {
                    "batch_size": 16,
                    "use_mmap": True
                },
                "auto_method": "tfidf"  # Use lightest method
            }
        elif total_gb < 8:  # Medium memory
            config["modeling"] = {
                "embeddings": {
                    "batch_size": 32,
                    "use_mmap": True
                },
                "auto_method": "simple_kmeans"
            }
        elif total_gb < 16:  # Good memory
            config["modeling"] = {
                "embeddings": {
                    "batch_size": 64
                },
                "auto_method": "simple_kmeans"
            }
        else:  # High memory
            config["modeling"] = {
                "embeddings": {
                    "batch_size": 128
                },
                "auto_method": "embedding_cluster"
            }
    except ImportError:
        # Default conservative values if psutil not available
        config["modeling"] = {
            "embeddings": {
                "batch_size": 32,
                "use_mmap": True
            },
            "auto_method": "simple_kmeans"
        }
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
            logger.info(f"GPU detected: {gpu_info['name']} with {gpu_info['memory']:.2f} GB memory")
            
            # Only use GPU if it has sufficient memory
            if gpu_info["memory"] > 2:
                config["modeling"]["embeddings"]["use_gpu"] = True
            else:
                config["modeling"]["embeddings"]["use_gpu"] = False
                logger.info("GPU has limited memory, defaulting to CPU for embeddings")
        else:
            config["modeling"]["embeddings"]["use_gpu"] = False
    except ImportError:
        config["modeling"]["embeddings"]["use_gpu"] = False
    
    return config


def get_optimal_topic_modeling_method(num_documents: int) -> str:
    """Determine the optimal topic modeling method based on dataset size.
    
    Parameters
    ----------
    num_documents : int
        Number of documents to process
        
    Returns
    -------
    str
        Recommended topic modeling method
    """
    if num_documents < 500:
        # Small dataset: full BERTopic is feasible
        return "bertopic"
    elif num_documents < 5000:
        # Medium dataset: embedding clustering is good
        return "embedding_cluster"
    elif num_documents < 20000:
        # Large dataset: simpler K-Means on embeddings
        return "simple_kmeans"
    elif num_documents < 100000:
        # Very large dataset: NMF is efficient
        return "nmf"
    else:
        # Extremely large dataset: TF-IDF with K-Means
        return "tfidf"


def load_user_config() -> Dict[str, Any]:
    """Load user configuration.
    
    Returns
    -------
    Dict[str, Any]
        User configuration
    """
    config_path = get_user_config_dir() / "config" / "user_config.yaml"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Error loading user config: {e}")
        return {}


def save_user_config(config: Dict[str, Any]) -> None:
    """Save user configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        User configuration to save
    """
    config_path = get_user_config_dir() / "config" / "user_config.yaml"
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"User configuration saved to {config_path}")
    except Exception as e:
        logger.warning(f"Error saving user config: {e}")


def update_user_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update user configuration with new values.
    
    Parameters
    ----------
    updates : Dict[str, Any]
        Configuration updates to apply
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration
    """
    config = load_user_config()
    
    # Update configuration (deep merge)
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    
    deep_update(config, updates)
    save_user_config(config)
    
    return config


def load_default_config() -> Dict[str, Any]:
    """Load default configuration from package resources.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration
    """
    # Look in several possible locations
    possible_paths = [
        Path(__file__).parent.parent / "default_config.yaml",  # package root
        Path(__file__).parent.parent.parent / "config" / "default_config.yaml",  # config directory
        Path(__file__).parent / "default_config.yaml",  # utils directory
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        return config
            except Exception as e:
                logger.warning(f"Error loading default config from {path}: {e}")
    
    # If no config found, return a minimal default
    logger.warning("No default configuration found. Using minimal defaults.")
    return {
        "preprocessing": {
            "normalization": {
                "lowercase": True,
                "remove_punctuation": True,
                "lemmatize": False
            }
        },
        "modeling": {
            "default_method": "simple_kmeans",
            "default_num_topics": 10,
            "embeddings": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "use_mmap": False,
                "use_gpu": False
            }
        },
        "visualization": {
            "default_plot_type": "embeddings"
        }
    }


def run_config_wizard() -> Dict[str, Any]:
    """Run interactive configuration wizard.
    
    Returns
    -------
    Dict[str, Any]
        User configuration based on wizard responses
    """
    config = {}
    
    print("\n" + "="*60)
    print("Welcome to Meno Configuration Wizard!")
    print("="*60)
    print("\nLet's set up your configuration for optimal performance.")
    print("(Press Enter to accept the default values shown in brackets)\n")
    
    # Get user's typical dataset size
    print("\n--- Dataset Configuration ---")
    dataset_sizes = {
        "s": "small (< 1,000 documents)",
        "m": "medium (1,000 - 10,000 documents)",
        "l": "large (10,000 - 100,000 documents)",
        "xl": "very large (> 100,000 documents)"
    }
    
    # Print options
    for key, desc in dataset_sizes.items():
        print(f"  {key}: {desc}")
    
    # Get user choice
    while True:
        dataset_choice = input("\nWhat size datasets do you typically work with? [m]: ").lower() or "m"
        if dataset_choice in dataset_sizes:
            break
        print("Invalid choice. Please select s, m, l, or xl.")
    
    # Configure based on dataset size
    modeling_config = {}
    if dataset_choice == "s":
        modeling_config["auto_method"] = "bertopic"
        modeling_config["default_method"] = "bertopic"
        modeling_config["default_num_topics"] = 10
    elif dataset_choice == "m":
        modeling_config["auto_method"] = "embedding_cluster"
        modeling_config["default_method"] = "embedding_cluster"
        modeling_config["default_num_topics"] = 15
    elif dataset_choice == "l":
        modeling_config["auto_method"] = "simple_kmeans"
        modeling_config["default_method"] = "simple_kmeans"
        modeling_config["default_num_topics"] = 20
    else:  # xl
        modeling_config["auto_method"] = "tfidf"
        modeling_config["default_method"] = "tfidf"
        modeling_config["default_num_topics"] = 30
    
    config["modeling"] = modeling_config
    
    # GPU configuration
    print("\n--- Hardware Configuration ---")
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            use_gpu = input("Use GPU for embedding computation? (faster but uses more memory) [Y/n]: ").lower() != "n"
        else:
            print("No GPU detected.")
            use_gpu = False
    except ImportError:
        print("PyTorch not installed or GPU not available.")
        use_gpu = False
    
    if "modeling" not in config:
        config["modeling"] = {}
    
    if "embeddings" not in config["modeling"]:
        config["modeling"]["embeddings"] = {}
    
    config["modeling"]["embeddings"]["use_gpu"] = use_gpu
    
    # Memory usage configuration
    print("\n--- Memory Usage Configuration ---")
    use_mmap = input("Use memory mapping for large datasets? (recommended for limited RAM) [Y/n]: ").lower() != "n"
    config["modeling"]["embeddings"]["use_mmap"] = use_mmap
    
    # Preprocessing configuration
    print("\n--- Preprocessing Configuration ---")
    normalize = input("Normalize text (lowercase, remove punctuation)? [Y/n]: ").lower() != "n"
    lemmatize = input("Lemmatize text (reduces words to root form, e.g., 'running' -> 'run')? [y/N]: ").lower() == "y"
    
    config["preprocessing"] = {
        "normalization": {
            "lowercase": normalize,
            "remove_punctuation": normalize,
            "lemmatize": lemmatize
        }
    }
    
    # Model selection
    print("\n--- Embedding Model Configuration ---")
    model_choices = {
        "1": "all-MiniLM-L6-v2 (fast, small, good quality)",
        "2": "all-mpnet-base-v2 (higher quality, slower)",
        "3": "all-distilroberta-v1 (balanced performance)",
        "4": "paraphrase-MiniLM-L3-v2 (fastest, smallest)"
    }
    
    # Print options
    for key, desc in model_choices.items():
        print(f"  {key}: {desc}")
    
    # Get user choice
    while True:
        model_choice = input("\nSelect embedding model [1]: ") or "1"
        if model_choice in model_choices:
            break
        print("Invalid choice. Please select 1, 2, 3, or 4.")
    
    # Map choices to model names
    model_names = {
        "1": "sentence-transformers/all-MiniLM-L6-v2",
        "2": "sentence-transformers/all-mpnet-base-v2",
        "3": "sentence-transformers/all-distilroberta-v1",
        "4": "sentence-transformers/paraphrase-MiniLM-L3-v2"
    }
    
    config["modeling"]["embeddings"]["model_name"] = model_names[model_choice]
    
    # Visualization preferences
    print("\n--- Visualization Configuration ---")
    vis_choices = {
        "1": "embeddings (interactive 2D/3D map of documents)",
        "2": "topics (bar chart of topic frequencies)",
        "3": "words (wordclouds for each topic)"
    }
    
    # Print options
    for key, desc in vis_choices.items():
        print(f"  {key}: {desc}")
    
    # Get user choice
    while True:
        vis_choice = input("\nDefault visualization type [1]: ") or "1"
        if vis_choice in vis_choices:
            break
        print("Invalid choice. Please select 1, 2, or 3.")
    
    # Map choices to visualization types
    vis_types = {
        "1": "embeddings",
        "2": "topics", 
        "3": "words"
    }
    
    config["visualization"] = {
        "default_plot_type": vis_types[vis_choice]
    }
    
    print("\n--- Configuration Complete ---")
    print("Your settings have been saved and will be used for future runs.")
    print("You can modify these settings at any time in ~/.meno/config/user_config.yaml\n")
    
    return config


def initialize_configuration() -> Dict[str, Any]:
    """Initialize configuration on first run.
    
    This function is called on the first run to create a user configuration
    based on system capabilities and user preferences.
    
    Returns
    -------
    Dict[str, Any]
        Initialized configuration
    """
    # Get system-based defaults
    system_config = analyze_system_capabilities()
    
    # Run interactive wizard if in interactive context
    if sys.stdin.isatty():  # Check if running in interactive terminal
        user_config = run_config_wizard()
        
        # Merge configs with user config taking precedence
        def deep_merge(d1, d2):
            """Merge d2 into d1, with d2 taking precedence."""
            result = d1.copy()
            for k, v in d2.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
        
        config = deep_merge(system_config, user_config)
    else:
        # Non-interactive: just use system config with defaults
        config = system_config
    
    # Save the config
    save_user_config(config)
    
    # Mark as initialized
    mark_initialized()
    
    return config


def merge_configs(*configs) -> Dict[str, Any]:
    """Merge multiple configurations.
    
    Later configurations take precedence over earlier ones.
    
    Parameters
    ----------
    *configs : Dict[str, Any]
        Configurations to merge
        
    Returns
    -------
    Dict[str, Any]
        Merged configuration
    """
    result = {}
    
    for config in configs:
        # Skip empty configs
        if not config:
            continue
            
        # Deep merge
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from various sources.
    
    This function loads and merges configurations from:
    1. Default package configuration
    2. User configuration
    3. Project configuration (if specified)
    
    Parameters
    ----------
    config_path : Optional[str], optional
        Path to project configuration file, by default None
        
    Returns
    -------
    Dict[str, Any]
        Merged configuration
    """
    # Check if this is first run
    first_run = is_first_run()
    
    # Load default configuration
    default_config = load_default_config()
    
    if first_run:
        # Initialize user configuration on first run
        user_config = initialize_configuration()
    else:
        # Load existing user configuration
        user_config = load_user_config()
    
    # Load project configuration if specified
    project_config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                project_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error loading project config from {config_path}: {e}")
    
    # Merge configurations (project > user > default)
    config = merge_configs(default_config, user_config, project_config)
    
    return config


def get_config_path(config_name: Optional[str] = None) -> Path:
    """Get path to a configuration file.
    
    Parameters
    ----------
    config_name : Optional[str], optional
        Name of the configuration file, by default None
        
    Returns
    -------
    Path
        Path to the configuration file
    """
    if config_name:
        # Check in project config directory
        project_path = Path.cwd() / "config" / f"{config_name}.yaml"
        if project_path.exists():
            return project_path
        
        # Check in user config directory
        user_path = get_user_config_dir() / "config" / f"{config_name}.yaml"
        if user_path.exists():
            return user_path
        
        # Check in package config directory
        package_paths = [
            Path(__file__).parent.parent / "config" / f"{config_name}.yaml",
            Path(__file__).parent.parent.parent / "config" / f"{config_name}.yaml",
        ]
        
        for path in package_paths:
            if path.exists():
                return path
        
        # Return user path even if it doesn't exist (for saving new config)
        return user_path
    else:
        # Return user config
        return get_user_config_dir() / "config" / "user_config.yaml"


def save_config_template(template_name: str, config: Dict[str, Any]) -> Path:
    """Save a configuration template.
    
    Parameters
    ----------
    template_name : str
        Name of the template
    config : Dict[str, Any]
        Configuration to save
        
    Returns
    -------
    Path
        Path to the saved template
    """
    # Ensure the filename has .yaml extension
    if not template_name.endswith('.yaml'):
        template_name += '.yaml'
    
    # Path in user config directory
    config_dir = get_user_config_dir() / "config" / "templates"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / template_name
    
    # Save the config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def list_config_templates() -> List[str]:
    """List available configuration templates.
    
    Returns
    -------
    List[str]
        List of template names
    """
    # Check user templates
    user_template_dir = get_user_config_dir() / "config" / "templates"
    if user_template_dir.exists():
        user_templates = [p.name for p in user_template_dir.glob("*.yaml")]
    else:
        user_templates = []
    
    # Check package templates
    package_template_dirs = [
        Path(__file__).parent.parent / "config" / "templates",
        Path(__file__).parent.parent.parent / "config" / "templates",
    ]
    
    package_templates = []
    for dir_path in package_template_dirs:
        if dir_path.exists():
            package_templates.extend([p.name for p in dir_path.glob("*.yaml")])
    
    # Combine and deduplicate
    all_templates = list(set(user_templates + package_templates))
    all_templates.sort()
    
    return all_templates