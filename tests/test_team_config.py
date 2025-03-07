"""Tests for the team configuration utilities in meno.utils.team_config."""

import json
import os
import yaml
import tempfile
from pathlib import Path
import pytest
from datetime import datetime

from meno.utils.team_config import (
    create_team_config,
    update_team_config,
    merge_team_configs,
    get_team_config_stats,
    compare_team_configs,
    export_team_acronyms,
    export_team_spelling_corrections,
    import_acronyms_from_file,
    import_spelling_corrections_from_file
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_acronyms():
    """Sample acronyms for testing."""
    return {
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "BERT": "Bidirectional Encoder Representations from Transformers",
        "CNN": "Convolutional Neural Network",
    }


@pytest.fixture
def sample_corrections():
    """Sample spelling corrections for testing."""
    return {
        "langauge": "language",
        "procesing": "processing",
        "artifical": "artificial",
        "intellegence": "intelligence",
        "learining": "learning",
    }


@pytest.fixture
def finance_acronyms():
    """Sample finance acronyms for testing."""
    return {
        "ROI": "Return on Investment",
        "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
        "P&L": "Profit and Loss",
        "YOY": "Year Over Year",
        "CAGR": "Compound Annual Growth Rate",
    }


@pytest.fixture
def finance_corrections():
    """Sample finance spelling corrections for testing."""
    return {
        "proffit": "profit",
        "expence": "expense",
        "ballance": "balance",
        "recievable": "receivable",
        "investmint": "investment",
    }


def test_create_team_config(temp_dir, sample_acronyms, sample_corrections):
    """Test creating a team configuration."""
    output_path = temp_dir / "ai_team_config.yaml"
    
    # Create configuration
    config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=output_path,
    )
    
    # Check that file was created
    assert output_path.exists()
    
    # Load the file and verify contents
    with open(output_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    
    # Check metadata
    assert loaded_config["metadata"]["team_name"] == "AI Research"
    assert "created" in loaded_config["metadata"]
    assert "last_modified" in loaded_config["metadata"]
    
    # Check acronyms
    for acronym, definition in sample_acronyms.items():
        assert loaded_config["preprocessing"]["acronyms"]["custom_mappings"][acronym] == definition
    
    # Check spelling corrections
    for misspelled, correction in sample_corrections.items():
        assert loaded_config["preprocessing"]["spelling"]["custom_dictionary"][misspelled] == correction


def test_update_team_config(temp_dir, sample_acronyms, sample_corrections):
    """Test updating a team configuration."""
    # First create a configuration
    output_path = temp_dir / "ai_team_config.yaml"
    config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=output_path,
    )
    
    # Record original timestamps
    with open(output_path, "r") as f:
        original_config = yaml.safe_load(f)
    original_created = original_config["metadata"]["created"]
    original_modified = original_config["metadata"]["last_modified"]
    
    # Update with new acronyms and corrections
    new_acronyms = {"RNN": "Recurrent Neural Network", "GAN": "Generative Adversarial Network"}
    new_corrections = {"recurrrent": "recurrent", "advesarial": "adversarial"}
    
    # Wait a bit to ensure timestamps are different
    import time
    time.sleep(1)
    
    # Update configuration
    config = update_team_config(
        config_path=output_path,
        acronyms=new_acronyms,
        spelling_corrections=new_corrections,
    )
    
    # Load updated configuration
    with open(output_path, "r") as f:
        updated_config = yaml.safe_load(f)
    
    # Check that timestamps were updated correctly
    assert updated_config["metadata"]["created"] == original_created
    assert updated_config["metadata"]["last_modified"] != original_modified
    
    # Check that new acronyms and corrections were added
    for acronym, definition in new_acronyms.items():
        assert updated_config["preprocessing"]["acronyms"]["custom_mappings"][acronym] == definition
    
    for misspelled, correction in new_corrections.items():
        assert updated_config["preprocessing"]["spelling"]["custom_dictionary"][misspelled] == correction
    
    # Check that original acronyms and corrections are still there
    for acronym, definition in sample_acronyms.items():
        assert updated_config["preprocessing"]["acronyms"]["custom_mappings"][acronym] == definition
    
    for misspelled, correction in sample_corrections.items():
        assert updated_config["preprocessing"]["spelling"]["custom_dictionary"][misspelled] == correction


def test_merge_team_configs(temp_dir, sample_acronyms, sample_corrections, finance_acronyms, finance_corrections):
    """Test merging team configurations."""
    # Create AI team config
    ai_config_path = temp_dir / "ai_team_config.yaml"
    ai_config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=ai_config_path,
    )
    
    # Create Finance team config
    finance_config_path = temp_dir / "finance_team_config.yaml"
    finance_config = create_team_config(
        team_name="Finance",
        acronyms=finance_acronyms,
        spelling_corrections=finance_corrections,
        output_path=finance_config_path,
    )
    
    # Merge configurations
    merged_path = temp_dir / "merged_config.yaml"
    merged_config = merge_team_configs(
        configs=[ai_config_path, finance_config_path],
        team_name="AI Finance",
        output_path=merged_path,
    )
    
    # Check that merged file was created
    assert merged_path.exists()
    
    # Load merged configuration
    with open(merged_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    
    # Check metadata
    assert loaded_config["metadata"]["team_name"] == "AI Finance"
    assert "created" in loaded_config["metadata"]
    assert "last_modified" in loaded_config["metadata"]
    assert "source_configs" in loaded_config["metadata"]
    assert len(loaded_config["metadata"]["source_configs"]) == 2
    
    # Check that all acronyms and corrections were merged
    all_acronyms = {**sample_acronyms, **finance_acronyms}
    all_corrections = {**sample_corrections, **finance_corrections}
    
    for acronym, definition in all_acronyms.items():
        assert loaded_config["preprocessing"]["acronyms"]["custom_mappings"][acronym] == definition
    
    for misspelled, correction in all_corrections.items():
        assert loaded_config["preprocessing"]["spelling"]["custom_dictionary"][misspelled] == correction


def test_get_team_config_stats(temp_dir, sample_acronyms, sample_corrections):
    """Test getting statistics for a team configuration."""
    # Create a configuration
    output_path = temp_dir / "ai_team_config.yaml"
    config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=output_path,
    )
    
    # Get statistics
    stats = get_team_config_stats(output_path)
    
    # Check statistics
    assert stats["team_name"] == "AI Research"
    assert "created" in stats
    assert "last_modified" in stats
    assert stats["acronym_count"] == len(sample_acronyms)
    assert stats["spelling_correction_count"] == len(sample_corrections)


def test_compare_team_configs(temp_dir, sample_acronyms, sample_corrections, finance_acronyms, finance_corrections):
    """Test comparing team configurations."""
    # Create AI team config
    ai_config_path = temp_dir / "ai_team_config.yaml"
    ai_config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=ai_config_path,
    )
    
    # Create Finance team config with some overlapping acronyms
    overlapping_finance_acronyms = finance_acronyms.copy()
    overlapping_finance_acronyms["AI"] = "Artificial Investment"  # Different definition for AI
    
    finance_config_path = temp_dir / "finance_team_config.yaml"
    finance_config = create_team_config(
        team_name="Finance",
        acronyms=overlapping_finance_acronyms,
        spelling_corrections=finance_corrections,
        output_path=finance_config_path,
    )
    
    # Compare configurations
    comparison = compare_team_configs(ai_config_path, finance_config_path)
    
    # Check comparison results
    assert comparison["team_names"]["config1"] == "AI Research"
    assert comparison["team_names"]["config2"] == "Finance"
    
    # Check acronym comparison
    assert "AI" in comparison["acronyms"]["differing_expansions"]
    assert len(comparison["acronyms"]["unique_to_config1"]) == len(sample_acronyms) - 1  # All except AI
    assert len(comparison["acronyms"]["unique_to_config2"]) == len(finance_acronyms)
    
    # Check spelling corrections comparison
    assert len(comparison["spelling_corrections"]["unique_to_config1"]) == len(sample_corrections)
    assert len(comparison["spelling_corrections"]["unique_to_config2"]) == len(finance_corrections)


def test_export_team_acronyms(temp_dir, sample_acronyms, sample_corrections):
    """Test exporting acronyms from a team configuration."""
    # Create a configuration
    config_path = temp_dir / "ai_team_config.yaml"
    config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=config_path,
    )
    
    # Export acronyms to JSON
    json_path = temp_dir / "acronyms.json"
    exported_acronyms = export_team_acronyms(
        config_path=config_path,
        output_format="json",
        output_path=json_path,
    )
    
    # Check that export file was created
    assert json_path.exists()
    
    # Load exported acronyms
    with open(json_path, "r") as f:
        loaded_acronyms = json.load(f)
    
    # Check that exported acronyms match
    assert loaded_acronyms == sample_acronyms
    
    # Export acronyms to YAML
    yaml_path = temp_dir / "acronyms.yaml"
    exported_acronyms = export_team_acronyms(
        config_path=config_path,
        output_format="yaml",
        output_path=yaml_path,
    )
    
    # Check that export file was created
    assert yaml_path.exists()
    
    # Load exported acronyms
    with open(yaml_path, "r") as f:
        loaded_acronyms = yaml.safe_load(f)
    
    # Check that exported acronyms match
    assert loaded_acronyms == sample_acronyms


def test_export_team_spelling_corrections(temp_dir, sample_acronyms, sample_corrections):
    """Test exporting spelling corrections from a team configuration."""
    # Create a configuration
    config_path = temp_dir / "ai_team_config.yaml"
    config = create_team_config(
        team_name="AI Research",
        acronyms=sample_acronyms,
        spelling_corrections=sample_corrections,
        output_path=config_path,
    )
    
    # Export corrections to JSON
    json_path = temp_dir / "corrections.json"
    exported_corrections = export_team_spelling_corrections(
        config_path=config_path,
        output_format="json",
        output_path=json_path,
    )
    
    # Check that export file was created
    assert json_path.exists()
    
    # Load exported corrections
    with open(json_path, "r") as f:
        loaded_corrections = json.load(f)
    
    # Check that exported corrections match
    assert loaded_corrections == sample_corrections
    
    # Export corrections to YAML
    yaml_path = temp_dir / "corrections.yaml"
    exported_corrections = export_team_spelling_corrections(
        config_path=config_path,
        output_format="yaml",
        output_path=yaml_path,
    )
    
    # Check that export file was created
    assert yaml_path.exists()
    
    # Load exported corrections
    with open(yaml_path, "r") as f:
        loaded_corrections = yaml.safe_load(f)
    
    # Check that exported corrections match
    assert loaded_corrections == sample_corrections


def test_import_acronyms_from_file(temp_dir, sample_acronyms):
    """Test importing acronyms from a file."""
    # Create a JSON file with acronyms
    json_path = temp_dir / "acronyms.json"
    with open(json_path, "w") as f:
        json.dump(sample_acronyms, f)
    
    # Import acronyms
    imported_acronyms = import_acronyms_from_file(json_path)
    
    # Check that imported acronyms match
    assert imported_acronyms == sample_acronyms
    
    # Create a YAML file with acronyms
    yaml_path = temp_dir / "acronyms.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_acronyms, f)
    
    # Import acronyms
    imported_acronyms = import_acronyms_from_file(yaml_path)
    
    # Check that imported acronyms match
    assert imported_acronyms == sample_acronyms


def test_import_spelling_corrections_from_file(temp_dir, sample_corrections):
    """Test importing spelling corrections from a file."""
    # Create a JSON file with corrections
    json_path = temp_dir / "corrections.json"
    with open(json_path, "w") as f:
        json.dump(sample_corrections, f)
    
    # Import corrections
    imported_corrections = import_spelling_corrections_from_file(json_path)
    
    # Check that imported corrections match
    assert imported_corrections == sample_corrections
    
    # Create a YAML file with corrections
    yaml_path = temp_dir / "corrections.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_corrections, f)
    
    # Import corrections
    imported_corrections = import_spelling_corrections_from_file(yaml_path)
    
    # Check that imported corrections match
    assert imported_corrections == sample_corrections