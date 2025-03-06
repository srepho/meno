"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st
import yaml

from meno.utils.config import load_config, merge_configs, MenoConfig


def test_default_config_loads():
    """Test that the default configuration loads correctly."""
    config = load_config(None)
    assert isinstance(config, MenoConfig)
    assert hasattr(config, "preprocessing")
    assert hasattr(config, "modeling")
    assert hasattr(config, "visualization")
    assert hasattr(config, "reporting")


def test_custom_config_loads():
    """Test that a custom configuration file loads correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
        preprocessing:
          normalization:
            lowercase: true
            remove_punctuation: false
            remove_numbers: true
            lemmatize: false
            language: "en"
        """
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)
        assert isinstance(config, MenoConfig)
        assert config.preprocessing.normalization.lowercase is True
        assert config.preprocessing.normalization.remove_punctuation is False
        assert config.preprocessing.normalization.remove_numbers is True
        assert config.preprocessing.normalization.lemmatize is False
        assert config.preprocessing.normalization.language == "en"
    finally:
        os.unlink(config_path)


def test_merge_configs():
    """Test that configuration merging works correctly."""
    base_config = load_config(None)
    
    # Create override config
    overrides = {
        "preprocessing": {
            "normalization": {
                "lowercase": False,
                "remove_numbers": True
            },
            "stopwords": {
                "additional": ["custom", "words"]
            }
        },
        "modeling": {
            "embeddings": {
                "model_name": "custom-model"
            }
        }
    }
    
    merged_config = merge_configs(base_config, overrides)
    
    # Check that overrides were applied
    assert merged_config.preprocessing.normalization.lowercase is False
    assert merged_config.preprocessing.normalization.remove_numbers is True
    assert "custom" in merged_config.preprocessing.stopwords.additional
    assert "words" in merged_config.preprocessing.stopwords.additional
    assert merged_config.modeling.embeddings.model_name == "custom-model"
    
    # Check that non-overridden values are maintained
    assert merged_config.preprocessing.normalization.remove_punctuation == base_config.preprocessing.normalization.remove_punctuation
    assert merged_config.preprocessing.normalization.lemmatize == base_config.preprocessing.normalization.lemmatize


@given(
    lowercase=st.booleans(),
    remove_punctuation=st.booleans(),
    remove_numbers=st.booleans(),
    lemmatize=st.booleans(),
    language=st.sampled_from(["en", "fr", "de", "es"]),
    model_name=st.text(min_size=3, max_size=30),
    batch_size=st.integers(min_value=1, max_value=128),
    n_neighbors=st.integers(min_value=2, max_value=100),
    min_dist=st.floats(min_value=0.01, max_value=1.0)
)
def test_config_property_types(
    lowercase, remove_punctuation, remove_numbers, lemmatize, language,
    model_name, batch_size, n_neighbors, min_dist
):
    """Test configuration with various property types using Hypothesis."""
    config_dict = {
        "preprocessing": {
            "normalization": {
                "lowercase": lowercase,
                "remove_punctuation": remove_punctuation,
                "remove_numbers": remove_numbers,
                "lemmatize": lemmatize,
                "language": language
            }
        },
        "modeling": {
            "embeddings": {
                "model_name": model_name,
                "batch_size": batch_size
            }
        },
        "visualization": {
            "umap": {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        # Check that properties are correctly set
        assert config.preprocessing.normalization.lowercase == lowercase
        assert config.preprocessing.normalization.remove_punctuation == remove_punctuation
        assert config.preprocessing.normalization.remove_numbers == remove_numbers
        assert config.preprocessing.normalization.lemmatize == lemmatize
        assert config.preprocessing.normalization.language == language
        
        assert config.modeling.embeddings.model_name == model_name
        assert config.modeling.embeddings.batch_size == batch_size
        
        assert config.visualization.umap.n_neighbors == n_neighbors
        assert config.visualization.umap.min_dist == min_dist
    finally:
        os.unlink(config_path)