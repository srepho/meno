"""Pytest configuration for meno tests."""

import pytest
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Optional
import tempfile

# Set fixed random seed for reproducibility
np.random.seed(42)


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(tests_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture
def create_temp_config_file():
    """Fixture to create a temporary configuration file."""
    def _create_config(config_content: str) -> str:
        """Create a temporary config file with the given content.
        
        Parameters
        ----------
        config_content : str
            YAML content to write to the file
            
        Returns
        -------
        str
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            return f.name
    
    return _create_config


@pytest.fixture
def cleanup_temp_files():
    """Fixture to keep track of and clean up temporary files."""
    temp_files = []
    
    def _register_file(file_path: str) -> str:
        """Register a file to be cleaned up after the test.
        
        Parameters
        ----------
        file_path : str
            Path to the file to be cleaned up
            
        Returns
        -------
        str
            The same file path, for chaining
        """
        temp_files.append(file_path)
        return file_path
    
    yield _register_file
    
    # Clean up all registered files
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.unlink(file_path)


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "preprocessing": {
            "normalization": {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_numbers": False,
                "lemmatize": True,
                "language": "en"
            },
            "stopwords": {
                "use_default": True,
                "additional": ["test", "example"]
            }
        },
        "modeling": {
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 32
            },
            "clustering": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "min_cluster_size": 10,
                "min_samples": 5
            },
            "topic_matching": {
                "threshold": 0.5,
                "assign_multiple": True,
                "max_topics_per_doc": 3
            }
        }
    }


@pytest.fixture
def create_test_documents(test_data_dir) -> List[str]:
    """Create a list of test documents."""
    documents = [
        "This is a test document about technology and computers.",
        "Healthcare and medicine are important for public health.",
        "Politics and government policies affect many aspects of life.",
        "Sports and exercise are good for physical and mental health.",
        "Education and learning are lifelong pursuits.",
        "Climate change is an important environmental issue.",
        "Financial markets and economics drive global trade.",
        "Music and art provide cultural enrichment.",
        "Food and nutrition are essential for well-being.",
        "Transportation and infrastructure connect communities."
    ]
    
    # Save to a test file for file-based tests
    test_file_path = os.path.join(test_data_dir, "test_documents.txt")
    with open(test_file_path, "w") as f:
        for doc in documents:
            f.write(f"{doc}\n")
    
    return documents


@pytest.fixture
def create_test_dataframe(create_test_documents) -> pd.DataFrame:
    """Create a test DataFrame with documents."""
    documents = create_test_documents
    return pd.DataFrame({
        "text": documents,
        "doc_id": [f"doc_{i}" for i in range(len(documents))]
    })