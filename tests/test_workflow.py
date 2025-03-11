"""Tests for the workflow module."""

import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import pytest
from unittest import mock
import re
from typing import Dict, List, Tuple

from meno.workflow import MenoWorkflow, create_workflow
from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector

# Sample data for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "text": [
            "The CEO and CFO met to discuss the AI implementation in our CRM system.",
            "HR dept is implementing a new PTO policy next month.",
            "IT team resolved the API issue affecting the CX system.",
            "Customer submitted a claim for their vehical accident on HWY 101.",
            "The CTO presented the ML strategy for improving cust retention.",
        ],
        "date": pd.date_range(start="2023-01-01", periods=5, freq="W"),
        "department": ["Executive", "HR", "IT", "Claims", "Technology"],
        "region": ["North", "South", "East", "West", "North"]
    })

# Test creation and initialization
def test_create_workflow():
    workflow = create_workflow()
    assert isinstance(workflow, MenoWorkflow)
    assert workflow.modeler is not None
    assert workflow.acronym_expander is not None
    assert workflow.spelling_corrector is not None
    assert workflow.documents is None
    assert workflow.text_column is None
    
    # Test with local_model_path parameter
    with mock.patch("meno.workflow.DocumentEmbedding") as mock_embed:
        mock_embed.return_value = mock.MagicMock()
        
        workflow = create_workflow(
            config_overrides={
                "modeling": {
                    "embeddings": {
                        "local_files_only": True
                    }
                }
            }
        )
        
        assert mock_embed.called
        
        # Test with explicit local_model_path
        mock_embed.reset_mock()
        workflow = create_workflow(
            config_overrides={
                "modeling": {
                    "embeddings": {
                        "local_model_path": "/mock/path"
                    }
                }
            }
        )
        
        assert mock_embed.called

# Test data loading
def test_load_data(sample_data):
    workflow = MenoWorkflow()
    
    # Test loading from DataFrame
    df = workflow.load_data(
        data=sample_data,
        text_column="text",
        time_column="date",
        geo_column="region",
        category_column="department"
    )
    
    assert df is not None
    assert len(df) == len(sample_data)
    assert workflow.text_column == "text"
    assert workflow.time_column == "date"
    assert workflow.geo_column == "region"
    assert workflow.category_column == "department"
    
    # Test validation of column names
    workflow = MenoWorkflow()
    with pytest.raises(ValueError):
        workflow.load_data(sample_data, text_column="non_existent_column")

# Test acronym detection
def test_detect_acronyms(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    acronyms = workflow.detect_acronyms(min_length=2, min_count=1)
    
    assert isinstance(acronyms, dict)
    assert len(acronyms) > 0
    assert "CEO" in acronyms
    assert "CFO" in acronyms
    assert "HR" in acronyms
    assert "AI" in acronyms

# Test acronym report generation
def test_generate_acronym_report(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        output_path = f.name
    
    try:
        report_path = workflow.generate_acronym_report(
            min_length=2,
            min_count=1,
            output_path=output_path,
            open_browser=False
        )
        
        assert os.path.exists(report_path)
        assert report_path == output_path
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert "<title>Meno Acronym Detection Report</title>" in content
            assert "CEO" in content
            assert "CFO" in content
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

# Test acronym expansion
def test_expand_acronyms(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    custom_mappings = {
        "CRM": "Customer Relationship Management",
        "HR": "Human Resources",
        "ML": "Machine Learning"
    }
    
    result = workflow.expand_acronyms(custom_mappings=custom_mappings)
    
    assert "Customer Relationship Management" in result["text"].iloc[0]
    assert "Human Resources" in result["text"].iloc[1]
    assert "Machine Learning" in result["text"].iloc[4]
    assert workflow.acronyms_expanded is True

# Test spelling detection
def test_detect_potential_misspellings(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    misspellings = workflow.detect_potential_misspellings(
        min_length=3,
        min_count=1
    )
    
    assert isinstance(misspellings, dict)
    
    # Check if 'vehical' is detected as misspelled
    contains_vehical = any(word == "vehical" for word in misspellings.keys())
    assert contains_vehical, "Should detect 'vehical' as misspelled"
    
    # Check if 'cust' is detected as misspelled
    contains_cust = any(word == "cust" for word in misspellings.keys())
    assert contains_cust, "Should detect 'cust' as misspelled"

# Test spelling correction
def test_correct_spelling(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    custom_corrections = {
        "vehical": "vehicle",
        "cust": "customer"
    }
    
    result = workflow.correct_spelling(custom_corrections=custom_corrections)
    
    # Check if corrections were applied
    assert "vehicle" in result["text"].iloc[3]
    assert "customer" in result["text"].iloc[4]
    assert workflow.spelling_corrected is True

# Test preprocessing
def test_preprocess_documents(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    
    processed = workflow.preprocess_documents(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True
    )
    
    assert processed is not None
    assert workflow.preprocessing_complete is True

# Test topic discovery - mock the underlying modeler
def test_discover_topics(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    workflow.preprocessing_complete = True
    
    # Mock the underlying modeler's discover_topics method
    with mock.patch.object(workflow.modeler, 'discover_topics') as mock_discover:
        mock_discover.return_value = pd.DataFrame({
            'topic': ['Topic_1', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_2'],
            'topic_probability': [0.85, 0.92, 0.78, 0.63, 0.71]
        })
        
        result = workflow.discover_topics(method="embedding_cluster", num_topics=3)
        
        assert result is not None
        assert len(result) == len(sample_data)
        assert 'topic' in result.columns
        assert 'topic_probability' in result.columns
        assert workflow.modeling_complete is True
        
        # Check if mock was called with correct parameters
        mock_discover.assert_called_once_with(
            method="embedding_cluster",
            num_topics=3
        )

# Test visualization - mock the underlying modeler's visualization methods
def test_visualize_topics(sample_data):
    workflow = MenoWorkflow()
    workflow.load_data(
        sample_data, 
        text_column="text",
        time_column="date",
        geo_column="region"
    )
    workflow.modeling_complete = True
    
    # Test visualization types
    with mock.patch.object(workflow.modeler, 'visualize_embeddings') as mock_viz_embeddings:
        mock_viz_embeddings.return_value = mock.MagicMock()
        
        # Test embeddings visualization
        viz = workflow.visualize_topics(plot_type="embeddings")
        assert viz is not None
        mock_viz_embeddings.assert_called_once()
    
    with mock.patch.object(workflow.modeler, 'visualize_topic_trends') as mock_viz_trends:
        mock_viz_trends.return_value = mock.MagicMock()
        
        # Test trend visualization
        viz = workflow.visualize_topics(plot_type="trends")
        assert viz is not None
        mock_viz_trends.assert_called_once_with(time_column="date")
    
    # Test with invalid plot type
    with pytest.raises(ValueError):
        workflow.visualize_topics(plot_type="invalid_type")

# Test the complete workflow
def test_run_complete_workflow(sample_data):
    workflow = MenoWorkflow()
    
    # Mock all the necessary methods
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        output_path = f.name
    
    try:
        with mock.patch.object(workflow, 'load_data') as mock_load:
            with mock.patch.object(workflow, 'expand_acronyms') as mock_expand:
                with mock.patch.object(workflow, 'correct_spelling') as mock_correct:
                    with mock.patch.object(workflow, 'preprocess_documents') as mock_preprocess:
                        with mock.patch.object(workflow, 'discover_topics') as mock_discover:
                            with mock.patch.object(workflow, 'generate_comprehensive_report') as mock_report:
                                mock_load.return_value = sample_data
                                mock_expand.return_value = sample_data
                                mock_correct.return_value = sample_data
                                mock_preprocess.return_value = sample_data
                                mock_discover.return_value = pd.DataFrame({
                                    'topic': ['Topic_1', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_2'],
                                    'topic_probability': [0.85, 0.92, 0.78, 0.63, 0.71]
                                })
                                mock_report.return_value = output_path
                                
                                # Run the complete workflow
                                result = workflow.run_complete_workflow(
                                    data=sample_data,
                                    text_column="text",
                                    time_column="date",
                                    geo_column="region",
                                    category_column="department",
                                    acronym_mappings={"CRM": "Customer Relationship Management"},
                                    spelling_corrections={"vehical": "vehicle"},
                                    modeling_method="embedding_cluster",
                                    num_topics=3,
                                    output_path=output_path,
                                    open_browser=False
                                )
                                
                                assert result == output_path
                                
                                # Verify each step was called with correct parameters
                                mock_load.assert_called_once()
                                mock_expand.assert_called_once_with(
                                    custom_mappings={"CRM": "Customer Relationship Management"}
                                )
                                mock_correct.assert_called_once_with(
                                    custom_corrections={"vehical": "vehicle"}
                                )
                                mock_preprocess.assert_called_once()
                                mock_discover.assert_called_once_with(
                                    method="embedding_cluster",
                                    num_topics=3
                                )
                                mock_report.assert_called_once_with(
                                    output_path=output_path,
                                    open_browser=False
                                )
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)