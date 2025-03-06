"""Functional tests for the configuration handling."""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

try:
    from meno.utils.config import load_config, merge_configs, MenoConfig
    ACTUAL_IMPORTS = True
except ImportError:
    ACTUAL_IMPORTS = False
    pytest.skip("Skipping functional config tests due to missing dependencies", allow_module_level=True)


@pytest.mark.skipif(not ACTUAL_IMPORTS, reason="Requires actual implementation")
class TestConfigFunctional:
    """Functional tests for configuration handling."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Create a sample configuration dictionary."""
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
                    "custom": ["test", "example"],
                    "keep": []
                }
            },
            "modeling": {
                "embeddings": {
                    "model_name": "all-MiniLM-L6-v2",
                    "batch_size": 32
                },
                "clustering": {
                    "algorithm": "kmeans",
                    "n_clusters": 5
                }
            }
        }
    
    @pytest.fixture
    def create_config_file(self, sample_config):
        """Create a temporary configuration file."""
        def _create_file():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(sample_config, f)
                return f.name
        
        file_path = _create_file()
        yield file_path
        
        # Clean up
        if os.path.exists(file_path):
            os.unlink(file_path)
    
    def test_load_config_from_file(self, create_config_file, sample_config):
        """Test loading configuration from a file."""
        config_path = create_config_file
        
        # Load config
        config = load_config(config_path)
        
        # Check that it's a MenoConfig instance
        assert isinstance(config, MenoConfig)
        
        # Check values match the sample config
        assert config.preprocessing.normalization.lowercase is sample_config["preprocessing"]["normalization"]["lowercase"]
        assert config.preprocessing.normalization.remove_punctuation is sample_config["preprocessing"]["normalization"]["remove_punctuation"]
        assert config.preprocessing.normalization.language == sample_config["preprocessing"]["normalization"]["language"]
        
        assert config.modeling.embeddings.model_name == sample_config["modeling"]["embeddings"]["model_name"]
        assert config.modeling.embeddings.batch_size == sample_config["modeling"]["embeddings"]["batch_size"]
        
        assert config.modeling.clustering.algorithm == sample_config["modeling"]["clustering"]["algorithm"]
        assert config.modeling.clustering.n_clusters == sample_config["modeling"]["clustering"]["n_clusters"]
        
        # Check stopwords
        assert config.preprocessing.stopwords.use_default is sample_config["preprocessing"]["stopwords"]["use_default"]
        assert list(config.preprocessing.stopwords.custom) == sample_config["preprocessing"]["stopwords"]["custom"]
    
    def test_load_default_config(self):
        """Test loading the default configuration."""
        # Load default config
        config = load_config(None)
        
        # Check that it's a MenoConfig instance
        assert isinstance(config, MenoConfig)
        
        # Check some default values
        assert config.preprocessing.normalization.lowercase is True
        assert config.preprocessing.normalization.language == "en"
        assert isinstance(config.modeling.embeddings.model_name, str)
        assert config.modeling.embeddings.batch_size > 0
    
    def test_merge_configs(self, sample_config):
        """Test merging configurations."""
        # Start with default config
        base_config = load_config(None)
        
        # Create overrides
        overrides = {
            "preprocessing": {
                "normalization": {
                    "lowercase": False,
                    "remove_numbers": True
                }
            },
            "modeling": {
                "embeddings": {
                    "model_name": "custom-model",
                    "batch_size": 64
                }
            }
        }
        
        # Merge configs
        merged = merge_configs(base_config, overrides)
        
        # Check merged values
        assert merged.preprocessing.normalization.lowercase is False  # Overridden
        assert merged.preprocessing.normalization.remove_numbers is True  # Overridden
        assert merged.preprocessing.normalization.remove_punctuation is True  # From base
        
        assert merged.modeling.embeddings.model_name == "custom-model"  # Overridden
        assert merged.modeling.embeddings.batch_size == 64  # Overridden
        
        # Original config should be unchanged
        assert base_config.preprocessing.normalization.lowercase is True
        assert base_config.preprocessing.normalization.remove_numbers is False
    
    def test_config_validation(self):
        """Test configuration validation checks."""
        # Create an invalid config
        invalid_config = {
            "preprocessing": {
                "normalization": {
                    "language": 123  # Should be a string
                }
            },
            "modeling": {
                "clustering": {
                    "algorithm": "invalid_algorithm",  # Invalid value
                    "n_clusters": -5  # Should be positive
                }
            }
        }
        
        # Load default config first
        base_config = load_config(None)
        
        # Merging should fail with validation errors
        with pytest.raises(Exception) as excinfo:
            merge_configs(base_config, invalid_config)
        
        # Check that the error message mentions the invalid fields
        error_str = str(excinfo.value)
        assert "language" in error_str or "algorithm" in error_str or "n_clusters" in error_str
    
    def test_config_serialization(self, sample_config):
        """Test config serialization to dict and yaml."""
        # Load sample config
        config = load_config(None)
        
        # Apply sample config overrides
        config = merge_configs(config, sample_config)
        
        # Convert to dict - handle both pydantic v1 and v2
        if hasattr(config, 'model_dump'):
            # Pydantic v2
            config_dict = config.model_dump()
        else:
            # Pydantic v1
            config_dict = config.dict()
        
        # Check it's a dictionary
        assert isinstance(config_dict, dict)
        assert "preprocessing" in config_dict
        assert "modeling" in config_dict
        
        # Convert to YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_file = f.name
            yaml.dump(config_dict, f)
        
        try:
            # Read back and check values
            with open(yaml_file, "r") as f:
                loaded_yaml = yaml.safe_load(f)
                
            assert loaded_yaml["preprocessing"]["normalization"]["lowercase"] is sample_config["preprocessing"]["normalization"]["lowercase"]
            assert loaded_yaml["modeling"]["embeddings"]["model_name"] == sample_config["modeling"]["embeddings"]["model_name"]
        finally:
            if os.path.exists(yaml_file):
                os.unlink(yaml_file)