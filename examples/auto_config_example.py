#!/usr/bin/env python3
"""Example showing automatic config file creation in Meno.

This example demonstrates how Meno automatically creates a configuration
file when one doesn't exist, making it easier for new users to get started.
"""

import pandas as pd
from pathlib import Path
import os
from meno.workflow import create_workflow
from meno.utils.config import create_default_config_file

# Sample data
data = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
    "Topic modeling is a type of statistical model for discovering abstract topics in a collection of documents.",
    "BERT is a transformer-based machine learning technique for natural language processing.",
    "GPT models are built on transformer architecture and trained on vast amounts of text data.",
    "Transformers have become the dominant architecture for large language models.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Neural networks are computing systems vaguely inspired by the biological neural networks in animal brains.",
    "Supervised learning is the machine learning task of learning a function that maps an input to an output.",
    "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in data."
]

# Create a DataFrame with the sample data
df = pd.DataFrame({"text": data})

def main():
    # Create a temporary directory for our example
    example_dir = Path("auto_config_example_output")
    os.makedirs(example_dir, exist_ok=True)
    
    print("\n--- Meno Auto Config Example ---\n")
    
    print("1. Creating a default config file explicitly:")
    # Example 1: Explicitly create a config file
    config_path = example_dir / "explicit_config.yaml"
    explicit_config_path = create_default_config_file(output_path=config_path)
    print(f"   Config file created at: {explicit_config_path}")
    
    print("\n2. Auto-creating a config file when initializing a workflow:")
    # Example 2: Create a workflow without providing a config file
    # This will automatically create a config in the current directory
    workflow = create_workflow()
    print(f"   Workflow initialized with auto-created config")
    
    print("\n3. Creating a workflow with a specific config path that doesn't exist yet:")
    # Example 3: Create a workflow with a specific config path
    specific_config_path = example_dir / "auto_created_config.yaml"
    # The config file doesn't exist yet, but will be created automatically
    workflow = create_workflow(config_path=specific_config_path)
    print(f"   Workflow initialized with config at: {specific_config_path}")
    
    print("\n4. Running a simplified workflow with auto-created config:")
    # Example 4: Run a simplified workflow
    workflow.load_data(df, text_column="text")
    workflow.preprocess_documents()
    results = workflow.discover_topics(method="embedding_cluster", num_topics=3)
    print(f"   Discovered topics: {results['topic'].unique()}")
    
    print("\n--- Example Complete ---")
    print(f"Config files and output saved to: {example_dir.absolute()}")

if __name__ == "__main__":
    main()