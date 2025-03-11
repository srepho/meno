"""Integration tests for lightweight models, visualizations, and web interface.

This file contains tests to ensure that all the new components work together correctly.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from meno.modeling.simple_models.lightweight_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel
)
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape, 
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)
# Mock the UnifiedTopicModeler since it has dependencies we don't need for testing
import sys
from unittest.mock import MagicMock

# Create a mock for the UnifiedTopicModeler
sys.modules['meno.modeling.unified_topic_modeling'] = MagicMock()
sys.modules['meno.modeling.unified_topic_modeling.UnifiedTopicModeler'] = MagicMock()

# Now import the web interface
from meno.web_interface import MenoWebApp


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        "Machine learning is a subfield of artificial intelligence that uses statistical techniques to enable computers to learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Neural networks are computing systems inspired by the biological neural networks in animal brains.",
        "Python is a popular programming language for data science and machine learning applications.",
        "TensorFlow and PyTorch are popular deep learning frameworks used to build neural networks.",
        "Natural language processing (NLP) enables computers to understand and interpret human language.",
        "Computer vision is a field of AI that enables computers to derive information from images and videos.",
        "Healthcare technology uses AI to improve diagnostics and patient care outcomes.",
        "Medical imaging uses computer vision techniques to analyze and interpret medical scans.",
        "Electronic health records (EHR) store patient data and medical history in digital format.",
        "Climate change refers to long-term shifts in global temperature and weather patterns.",
        "Renewable energy sources like solar and wind power help reduce carbon emissions.",
        "Sustainable development aims to meet human needs while preserving the environment.",
        "Conservation efforts focus on protecting biodiversity and natural habitats.",
        "Electric vehicles reduce reliance on fossil fuels and lower carbon emissions."
    ]


@pytest.fixture
def sample_dataframe(sample_documents):
    """Create a sample DataFrame with documents for testing."""
    return pd.DataFrame({
        "text": sample_documents,
        "doc_id": [f"doc_{i}" for i in range(len(sample_documents))]
    })


@pytest.fixture
def fitted_models(sample_documents, monkeypatch):
    """Create a list of fitted models for testing using mocks for embeddings."""
    # Create mock embedding function
    def mock_embed_documents(self, docs):
        """Mock embedding function that returns random embeddings."""
        return np.random.random((len(docs), 384))  # Standard embedding size
    
    # Apply the mock to the DocumentEmbedding class
    monkeypatch.setattr('meno.modeling.embeddings.DocumentEmbedding.embed_documents', mock_embed_documents)
    
    # Create and fit models
    simple_model = SimpleTopicModel(num_topics=3, random_state=42)
    simple_model.fit(sample_documents)
    
    tfidf_model = TFIDFTopicModel(num_topics=3, random_state=42)
    tfidf_model.fit(sample_documents)
    
    nmf_model = NMFTopicModel(num_topics=3, random_state=42)
    nmf_model.fit(sample_documents)
    
    lsa_model = LSATopicModel(num_topics=3, random_state=42)
    lsa_model.fit(sample_documents)
    
    return {
        "simple": simple_model,
        "tfidf": tfidf_model,
        "nmf": nmf_model,
        "lsa": lsa_model
    }


@pytest.fixture
def mock_web_app():
    """Create a mock web app for testing."""
    app = MagicMock()
    app.port = 8050
    app.debug = False
    app.temp_dir = tempfile.mkdtemp(prefix="meno_test_")
    app.app = MagicMock()
    app.layout = MagicMock()
    return app


class TestIntegratedComponents:
    """Tests for integration between lightweight models, visualizations, and web interface."""
    
    def test_model_initialization(self):
        """Test initialization of all lightweight models."""
        simple_model = SimpleTopicModel()
        tfidf_model = TFIDFTopicModel()
        nmf_model = NMFTopicModel()
        lsa_model = LSATopicModel()
        
        # Verify models have correct interfaces
        for model in [simple_model, tfidf_model, nmf_model, lsa_model]:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'transform')
            assert hasattr(model, 'get_topic_info')
            assert hasattr(model, 'get_document_info')
            assert hasattr(model, 'get_topic')
            assert hasattr(model, 'visualize_topics')
    
    def test_model_fitting(self, sample_documents, monkeypatch):
        """Test fitting all lightweight models."""
        # Create mock embedding function
        def mock_embed_documents(self, docs):
            """Mock embedding function that returns random embeddings."""
            return np.random.random((len(docs), 384))  # Standard embedding size
        
        # Apply the mock to the DocumentEmbedding class
        monkeypatch.setattr('meno.modeling.embeddings.DocumentEmbedding.embed_documents', mock_embed_documents)
        
        # Create models for testing
        models = [
            SimpleTopicModel(num_topics=3),
            TFIDFTopicModel(num_topics=3),
            NMFTopicModel(num_topics=3),
            LSATopicModel(num_topics=3)
        ]
        
        for model in models:
            # Fit model
            fitted_model = model.fit(sample_documents)
            
            # Verify the model is correctly flagged as fitted
            assert fitted_model.is_fitted
            
            # Verify topic information
            topic_info = fitted_model.get_topic_info()
            assert isinstance(topic_info, pd.DataFrame)
            assert "Topic" in topic_info.columns
            
            # Get document info
            doc_info = fitted_model.get_document_info()
            assert len(doc_info) == len(sample_documents)
    
    def test_model_comparison_visualization(self, fitted_models, sample_documents):
        """Test model comparison visualization with multiple models."""
        # Get all fitted models
        models = list(fitted_models.values())
        model_names = list(fitted_models.keys())
        
        # Create model comparison visualization
        fig = plot_model_comparison(
            document_lists=[sample_documents] * len(models),
            model_names=model_names,
            models=models
        )
        
        # Verify figure was created
        assert fig is not None
    
    def test_topic_landscape_visualization(self, fitted_models, sample_documents):
        """Test topic landscape visualization with different models."""
        # Test with each model type
        for model_name, model in fitted_models.items():
            fig = plot_topic_landscape(
                model=model,
                documents=sample_documents,
                title=f"{model_name.upper()} Topic Landscape"
            )
            
            # Verify figure was created
            assert fig is not None
    
    def test_multi_topic_heatmap(self, fitted_models, sample_documents):
        """Test multi-topic heatmap with different models."""
        # Select two models for comparison
        models_to_compare = ["simple", "tfidf"]
        model_list = [fitted_models[name] for name in models_to_compare]
        
        # Create heatmap
        fig = plot_multi_topic_heatmap(
            models=model_list,
            model_names=models_to_compare,
            document_lists=[sample_documents] * len(models_to_compare)
        )
        
        # Verify figure was created
        assert fig is not None
    
    def test_document_analysis(self, fitted_models, sample_documents):
        """Test document analysis with different models."""
        # Test with each model type
        for model_name, model in fitted_models.items():
            fig = plot_comparative_document_analysis(
                model=model,
                documents=sample_documents[:10],  # Use first 10 documents
                title=f"{model_name.upper()} Document Analysis"
            )
            
            # Verify figure was created
            assert fig is not None
    
    @patch('dash.Dash')
    def test_web_app_initialization(self, mock_dash):
        """Test web app initialization."""
        # Initialize web app
        app = MenoWebApp(port=8051, debug=False)
        
        # Verify app was initialized
        assert app.port == 8051
        assert app.debug is False
        assert hasattr(app, 'app')
        assert hasattr(app, 'temp_dir')
    
    @patch('meno.web_interface.MenoWebApp')
    def test_web_app_launch(self, mock_web_app_class):
        """Test web app launch function."""
        from meno.web_interface import launch_web_interface
        
        # Configure mock
        mock_app = MagicMock()
        mock_web_app_class.return_value = mock_app
        
        # Test launch function
        launch_web_interface(port=8052, debug=True)
        
        # Verify app was created with correct parameters
        mock_web_app_class.assert_called_once_with(port=8052, debug=True)
        mock_app.run.assert_called_once()
        mock_app.cleanup.assert_called_once()
    
    def test_model_integration_with_visualization(self, fitted_models, sample_documents):
        """Test integration between models and visualizations."""
        # Use one model for testing
        model = fitted_models["simple"]
        
        # 1. Get topic info
        topic_info = model.get_topic_info()
        
        # 2. Get document topic assignments
        doc_info = model.get_document_info()
        
        # 3. Create visualization
        fig = model.visualize_topics()
        assert fig is not None
        
        # 4. Create topic landscape
        landscape_fig = plot_topic_landscape(model, sample_documents)
        assert landscape_fig is not None
        
        # 5. Create document analysis
        doc_fig = plot_comparative_document_analysis(model, sample_documents)
        assert doc_fig is not None
    
    def test_model_serialization(self, fitted_models):
        """Test model serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_name, model in fitted_models.items():
                # Create save path
                save_path = Path(tmp_dir) / model_name
                
                # Save model
                model.save(save_path)
                
                # Verify files were created
                assert (save_path / "model_data.pkl").exists()
                
                # Load model
                if model_name == "simple":
                    loaded_model = SimpleTopicModel.load(save_path)
                elif model_name == "tfidf":
                    loaded_model = TFIDFTopicModel.load(save_path)
                elif model_name == "nmf":
                    loaded_model = NMFTopicModel.load(save_path)
                elif model_name == "lsa":
                    loaded_model = LSATopicModel.load(save_path)
                
                # Verify model was loaded correctly
                assert loaded_model.is_fitted
                assert loaded_model.num_topics == model.num_topics
                assert loaded_model.topics == model.topics
    
    @pytest.mark.parametrize("model_name", ["simple", "tfidf", "nmf", "lsa"])
    def test_model_transform(self, fitted_models, sample_documents, model_name):
        """Test transform method on new documents."""
        model = fitted_models[model_name]
        
        # New test documents
        new_docs = [
            "Artificial intelligence is transforming many industries.",
            "Sustainable practices are important for environmental protection."
        ]
        
        # Transform documents
        result = model.transform(new_docs)
        
        # For SimpleTopicModel and TFIDFTopicModel, result should be a tuple with assignments and matrix
        if model_name in ["simple", "tfidf"]:
            assignments, doc_topic_matrix = result
            assert len(assignments) == len(new_docs)
            assert doc_topic_matrix.shape == (len(new_docs), model.num_topics)
        # For NMF and LSA, result should be just the doc-topic matrix
        else:
            doc_topic_matrix = result
            assert doc_topic_matrix.shape == (len(new_docs), model.num_topics)