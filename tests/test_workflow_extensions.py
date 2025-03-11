"""Tests for the new workflow extension methods.

This test file focuses on the get_preprocessed_data and set_topic_assignments
methods in the MenoWorkflow class.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from meno.workflow import MenoWorkflow


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


@pytest.fixture
def topic_assignments():
    return pd.DataFrame({
        "topic": [0, 1, 2, 1, 0],
        "topic_probability": [0.85, 0.92, 0.78, 0.63, 0.71]
    })


class TestGetPreprocessedData:
    """Tests for the get_preprocessed_data method."""

    def test_get_preprocessed_data_success(self, sample_data):
        """Test successful retrieval of preprocessed data."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        # Mock the preprocessing to set the flag
        with patch.object(workflow.modeler, 'preprocess', return_value=sample_data):
            workflow.preprocess_documents()
            # Mock the preprocessed data in modeler
            workflow.modeler.documents = sample_data.copy()
            workflow.modeler.documents["preprocessed_text"] = [
                "ceo cfo meet discuss ai implementation crm system",
                "hr dept implement new pto policy next month",
                "it team resolve api issue affect cx system",
                "customer submit claim vehical accident hwy",
                "cto present ml strategy improve cust retention"
            ]
            
            # Execute
            result = workflow.get_preprocessed_data()
            
            # Verify
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_data)
            assert "preprocessed_text" in result.columns
            assert result.equals(workflow.modeler.documents)

    def test_get_preprocessed_data_not_preprocessed(self, sample_data):
        """Test error when trying to get preprocessed data before preprocessing."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        workflow.preprocessing_complete = False
        
        # Execute & Verify
        with pytest.raises(ValueError, match="No preprocessing has been performed"):
            workflow.get_preprocessed_data()

    def test_get_preprocessed_data_after_preprocessing(self, sample_data):
        """Test getting preprocessed data after actual preprocessing."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        
        # Execute actual preprocessing
        workflow.preprocess_documents()
        result = workflow.get_preprocessed_data()
        
        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert "preprocessed_text" in result.columns


class TestSetTopicAssignments:
    """Tests for the set_topic_assignments method."""

    def test_set_topic_assignments_success(self, sample_data, topic_assignments):
        """Test successful setting of topic assignments."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        # Ensure documents are set in modeler
        workflow.preprocess_documents()
        
        # Set the same index for topic assignments
        topic_assignments.index = workflow.modeler.documents.index
        
        # Execute
        workflow.set_topic_assignments(topic_assignments)
        
        # Verify
        assert workflow.modeling_complete is True
        assert "topic" in workflow.modeler.documents.columns
        assert "topic_probability" in workflow.modeler.documents.columns
        assert workflow.modeler.topic_assignments is topic_assignments
        assert workflow.modeler.documents["topic"].equals(topic_assignments["topic"])
        assert workflow.modeler.documents["topic_probability"].equals(topic_assignments["topic_probability"])

    def test_set_topic_assignments_no_documents(self, topic_assignments):
        """Test error when no documents are loaded."""
        # Setup
        workflow = MenoWorkflow()
        
        # Execute & Verify
        with pytest.raises(ValueError, match="No documents loaded"):
            workflow.set_topic_assignments(topic_assignments)

    def test_set_topic_assignments_missing_topic_column(self, sample_data):
        """Test error when topic column is missing."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        workflow.preprocess_documents()
        
        # Create assignments without topic column
        bad_assignments = pd.DataFrame({
            "probability": [0.85, 0.92, 0.78, 0.63, 0.71]
        })
        bad_assignments.index = workflow.modeler.documents.index
        
        # Execute & Verify
        with pytest.raises(ValueError, match="must contain a 'topic' column"):
            workflow.set_topic_assignments(bad_assignments)

    def test_set_topic_assignments_index_mismatch(self, sample_data, topic_assignments):
        """Test error when index doesn't match."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        workflow.preprocess_documents()
        
        # Assign different index to create mismatch
        topic_assignments.index = pd.RangeIndex(10, 15)
        
        # Execute & Verify
        with pytest.raises(ValueError, match="index does not match documents index"):
            workflow.set_topic_assignments(topic_assignments)

    def test_set_topic_assignments_without_probability(self, sample_data):
        """Test setting topic assignments without probability column."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        workflow.preprocess_documents()
        
        # Create assignments with only topic column
        topic_only = pd.DataFrame({
            "topic": [0, 1, 2, 1, 0]
        })
        topic_only.index = workflow.modeler.documents.index
        
        # Execute
        workflow.set_topic_assignments(topic_only)
        
        # Verify
        assert workflow.modeling_complete is True
        assert "topic" in workflow.modeler.documents.columns
        assert "topic_probability" not in workflow.modeler.documents.columns
        assert workflow.modeler.topic_assignments is topic_only
        assert workflow.modeler.documents["topic"].equals(topic_only["topic"])

    def test_integration_with_external_model(self, sample_data):
        """Test integration of external model topics with the workflow."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text")
        workflow.preprocess_documents()
        
        # Create simulated BERTopic results
        external_topics = pd.DataFrame({
            "topic": [-1, 0, 1, 0, 1],  # -1 represents outlier in BERTopic
            "topic_probability": [0.55, 0.92, 0.88, 0.75, 0.83]
        })
        external_topics.index = workflow.modeler.documents.index
        
        # Execute
        workflow.set_topic_assignments(external_topics)
        
        # Verify
        assert workflow.modeling_complete is True
        assert "topic" in workflow.modeler.documents.columns
        assert external_topics["topic"].equals(workflow.modeler.documents["topic"])
        
        # External topics should be preserved as-is, even with outlier (-1) topics
        assert -1 in workflow.modeler.documents["topic"].values

    def test_visualize_after_set_topic_assignments(self, sample_data):
        """Test that visualizations work after setting external topic assignments."""
        # Setup
        workflow = MenoWorkflow()
        workflow.load_data(sample_data, text_column="text", time_column="date")
        workflow.preprocess_documents()
        
        # Create and set external topics
        external_topics = pd.DataFrame({
            "topic": [0, 1, 2, 1, 0],
            "topic_probability": [0.85, 0.92, 0.78, 0.63, 0.71]
        })
        external_topics.index = workflow.modeler.documents.index
        workflow.set_topic_assignments(external_topics)
        
        # Mock the visualization methods to verify they're called correctly
        with patch.object(workflow.modeler, 'visualize_topic_distribution') as mock_viz:
            mock_viz.return_value = MagicMock()
            
            # Execute
            workflow.visualize_topics(plot_type="distribution")
            
            # Verify
            assert mock_viz.called


def test_end_to_end_external_topics_workflow(sample_data):
    """Test an end-to-end workflow using external topic assignments."""
    # Setup
    workflow = MenoWorkflow()
    workflow.load_data(sample_data, text_column="text")
    workflow.preprocess_documents()
    
    # Get preprocessed data
    preprocessed = workflow.get_preprocessed_data()
    assert "preprocessed_text" in preprocessed.columns
    
    # Simulate external model processing (e.g., from BERTopic)
    # The external model would use this preprocessed data
    
    # Set external topic assignments
    external_topics = pd.DataFrame({
        "topic": [0, 1, 0, 2, 1],
        "topic_probability": [0.91, 0.87, 0.83, 0.79, 0.88]
    })
    external_topics.index = preprocessed.index
    
    # Set topics in workflow
    workflow.set_topic_assignments(external_topics)
    
    # Verify topics are correctly set
    assert workflow.modeling_complete is True
    
    # Mock report generation to avoid actual file creation
    with patch.object(workflow.modeler, 'generate_report') as mock_report:
        mock_report.return_value = "mock_report.html"
        
        # Generate report using external topics
        report_path = workflow.generate_comprehensive_report(open_browser=False)
        
        # Verify report generation was called
        assert mock_report.called
        assert report_path == "mock_report.html"