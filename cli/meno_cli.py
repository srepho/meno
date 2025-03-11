"""Main command-line interface for Meno.

This module provides a CLI for running Meno topic modeling on text data.
"""

import argparse
import sys
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meno.workflow import Workflow
from meno.utils import config_manager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("meno-cli")


def load_data(file_path: str, text_column: str) -> pd.DataFrame:
    """Load data from a file.
    
    Parameters
    ----------
    file_path : str
        Path to the data file
    text_column : str
        Name of the column containing text data
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    # Check file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Determine file type from extension
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            # Try to read as text file with one document per line
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({text_column: texts})
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)
    
    # Verify text column exists
    if text_column not in df.columns:
        available_columns = ", ".join(df.columns)
        logger.error(f"Column '{text_column}' not found in file. Available columns: {available_columns}")
        sys.exit(1)
    
    return df


def analyze_command(args):
    """Run the analyze command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Load data
    logger.info(f"Loading data from {args.file}...")
    data = load_data(args.file, args.text_column)
    logger.info(f"Loaded {len(data)} documents")
    
    # Create workflow
    config_path = args.config if args.config else None
    workflow = Workflow(config_path=config_path)
    
    # Determine modeling method if "auto"
    modeling_method = args.method
    if modeling_method == "auto":
        modeling_method = config_manager.get_optimal_topic_modeling_method(len(data))
        logger.info(f"Auto-selected modeling method: {modeling_method}")
    
    # Adjust other parameters
    kwargs = {}
    if args.num_topics:
        kwargs["num_topics"] = args.num_topics
    
    # Run workflow
    try:
        # Determine paths
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.file) or os.getcwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{args.name}.html"
        
        # Log settings
        logger.info(f"Analysis settings:")
        logger.info(f"  - Method: {modeling_method}")
        logger.info(f"  - Number of topics: {kwargs.get('num_topics', 'auto')}")
        logger.info(f"  - Output: {output_path}")
        
        # Run workflow
        workflow.run_complete_workflow(
            data=data,
            text_column=args.text_column,
            modeling_method=modeling_method,
            output_path=str(output_path),
            open_browser=args.open_browser,
            **kwargs
        )
        
        logger.info(f"Analysis complete. Report saved to {output_path}")
        
        if args.open_browser:
            logger.info("Opening report in browser...")
        else:
            logger.info(f"To view the report, open the file in a web browser: {output_path}")
    
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def configure_command(args):
    """Run the configure command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Force reconfiguration if requested
    if args.reset:
        # Delete the initialization marker
        config_marker = config_manager.get_user_config_dir() / "initialized"
        if config_marker.exists():
            config_marker.unlink()
            logger.info("Configuration reset. Running wizard...")
    
    # Create new configuration
    config = config_manager.initialize_configuration()
    logger.info(f"Configuration complete. Saved to {config_manager.get_config_path()}")


def list_methods_command(args):
    """List available topic modeling methods.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    methods = [
        {
            "name": "bertopic",
            "description": "Full BERTopic implementation (best quality, heaviest)",
            "dataset_size": "Small (<1000 documents)",
            "dependencies": "umap-learn, hdbscan, bertopic"
        },
        {
            "name": "embedding_cluster",
            "description": "Embedding-based clustering with optimizations",
            "dataset_size": "Medium (1000-10000 documents)",
            "dependencies": "umap-learn, hdbscan"
        },
        {
            "name": "simple_kmeans",
            "description": "K-Means clustering on document embeddings",
            "dataset_size": "Large (10000-50000 documents)",
            "dependencies": "scikit-learn"
        },
        {
            "name": "nmf",
            "description": "Non-negative Matrix Factorization topic modeling",
            "dataset_size": "Large (10000-100000 documents)",
            "dependencies": "scikit-learn"
        },
        {
            "name": "lsa",
            "description": "Latent Semantic Analysis (LSA/LSI) topic modeling",
            "dataset_size": "Large (10000-100000 documents)",
            "dependencies": "scikit-learn"
        },
        {
            "name": "tfidf",
            "description": "TF-IDF vectorization with K-Means clustering",
            "dataset_size": "Very large (>100000 documents)",
            "dependencies": "scikit-learn"
        },
        {
            "name": "auto",
            "description": "Automatically select method based on dataset size",
            "dataset_size": "Any",
            "dependencies": "Varies by selection"
        }
    ]
    
    # Print as table
    print("\nAvailable Topic Modeling Methods:")
    print("="*80)
    print(f"{'Name':<15} {'Description':<40} {'Dataset Size':<25} {'Dependencies'}")
    print("-"*80)
    
    for method in methods:
        print(f"{method['name']:<15} {method['description']:<40} {method['dataset_size']:<25} {method['dependencies']}")
    
    print("\nUsage: meno analyze --method <method_name> ...")
    print("\nNote: 'auto' will select the appropriate method based on your dataset size.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Meno: Topic Modeling for Text Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze text data and discover topics"
    )
    analyze_parser.add_argument(
        "--file", "-f", required=True,
        help="Path to input file (CSV, Excel, JSON, TSV, or TXT)"
    )
    analyze_parser.add_argument(
        "--text-column", "-t", default="text",
        help="Name of column containing text data"
    )
    analyze_parser.add_argument(
        "--method", "-m", default="auto",
        choices=["auto", "bertopic", "embedding_cluster", "simple_kmeans", "nmf", "lsa", "tfidf"],
        help="Topic modeling method to use"
    )
    analyze_parser.add_argument(
        "--num-topics", "-n", type=int,
        help="Number of topics to extract (default: auto-determined)"
    )
    analyze_parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save output files (default: same as input file)"
    )
    analyze_parser.add_argument(
        "--name", default="meno_report",
        help="Base name for output files"
    )
    analyze_parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    analyze_parser.add_argument(
        "--open-browser", "-b", action="store_true",
        help="Open report in browser when complete"
    )
    analyze_parser.set_defaults(func=analyze_command)
    
    # Configure command
    configure_parser = subparsers.add_parser(
        "configure", help="Configure Meno settings"
    )
    configure_parser.add_argument(
        "--reset", "-r", action="store_true",
        help="Reset configuration to defaults"
    )
    configure_parser.set_defaults(func=configure_command)
    
    # List methods command
    methods_parser = subparsers.add_parser(
        "methods", help="List available topic modeling methods"
    )
    methods_parser.set_defaults(func=list_methods_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()