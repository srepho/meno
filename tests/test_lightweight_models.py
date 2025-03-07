"""Tests for the lightweight topic models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from meno.modeling.simple_models import (
    SimpleTopicModel,
    TFIDFTopicModel, 
    NMFTopicModel,
    LSATopicModel
)
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler


@pytest.fixture
def sample_documents():
    """Create a sample dataset for testing."""
    return [
        "Customer service was excellent and the product arrived on time.",
        "The package was damaged during shipping but customer service resolved the issue.",
        "Product quality is outstanding and exceeded my expectations.",
        "The software has a steep learning curve but powerful features.",
        "Technical support was helpful in resolving my installation issues.",
        "User interface is intuitive and easy to navigate.",
        "The documentation lacks examples and could be improved.",
        "Performance is excellent even with large datasets.",
        "Pricing is competitive compared to similar products.",
        "Regular updates keep adding valuable new features.",
        "The mobile app lacks some features available in the desktop version.",
        "Setup was straightforward and took only a few minutes.",
        "The product is reliable and hasn't crashed in months of use.",
        "Customer service response time could be improved.",
        "Training materials are comprehensive and well-structured.",
        "The hardware integration works seamlessly with our existing systems.",
        "Battery life is impressive compared to previous models.",
        "The AI features are innovative but sometimes unpredictable.",
        "Security features are robust and meet our compliance requirements.",
        "The community forum is active and helpful for troubleshooting."
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock document embedding model."""
    model = MagicMock()
    model.embed_documents.return_value = np.random.rand(20, 384)  # Random embeddings
    return model


class TestSimpleTopicModel:
    """Test the SimpleTopicModel class."""
    
    def test_init(self):
        """Test initialization of SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=5, random_state=42)
        
        assert model.num_topics == 5
        assert model.random_state == 42
        assert model.is_fitted is False
        assert model.topics == {}
        assert model.topic_words == {}
        assert model.topic_sizes == {}
    
    def test_fit(self, sample_documents, mock_embedding_model):
        """Test fitting the SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=3, embedding_model=mock_embedding_model, random_state=42)
        fitted_model = model.fit(sample_documents)
        
        # Verify the model is fitted
        assert fitted_model is model
        assert model.is_fitted is True
        
        # Verify embedding model was called
        mock_embedding_model.embed_documents.assert_called_once_with(sample_documents)
        
        # Verify topics were created
        assert len(model.topics) == 3
        assert len(model.topic_words) == 3
        assert len(model.topic_sizes) == 3
        
        # Verify clusters were assigned
        assert hasattr(model, 'clusters')
        assert len(model.clusters) == len(sample_documents)
    
    def test_transform(self, sample_documents, mock_embedding_model):
        """Test transforming documents with SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=3, embedding_model=mock_embedding_model, random_state=42)
        model.fit(sample_documents)
        
        # Reset mock to verify transform call
        mock_embedding_model.embed_documents.reset_mock()
        
        # Transform new documents
        result = model.transform(sample_documents[:2])
        
        # Verify embedding model was called
        mock_embedding_model.embed_documents.assert_called_once_with(sample_documents[:2])
        
        # Verify result shape
        assert result.shape == (2, 3)
        
        # Each document should be assigned to exactly one topic
        assert np.sum(result, axis=1).tolist() == [1.0, 1.0]
    
    def test_fit_transform(self, sample_documents, mock_embedding_model):
        """Test fit_transform with SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=3, embedding_model=mock_embedding_model, random_state=42)
        
        # Call fit_transform
        result = model.fit_transform(sample_documents)
        
        # Verify model is fitted
        assert model.is_fitted is True
        
        # Verify result shape
        assert result.shape == (len(sample_documents), 3)
        
        # Each document should be assigned to exactly one topic
        assert np.all(np.sum(result, axis=1) == 1.0)
    
    def test_get_topic_info(self, sample_documents, mock_embedding_model):
        """Test getting topic info from SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=3, embedding_model=mock_embedding_model, random_state=42)
        model.fit(sample_documents)
        
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(topic_info, pd.DataFrame)
        assert "Topic" in topic_info.columns
        assert "Name" in topic_info.columns
        assert "Size" in topic_info.columns
        assert "Words" in topic_info.columns
        
        # Verify all topics are present
        assert len(topic_info) == 3
        assert set(topic_info["Topic"].tolist()) == {0, 1, 2}
    
    def test_get_document_info(self, sample_documents, mock_embedding_model):
        """Test getting document info from SimpleTopicModel."""
        model = SimpleTopicModel(num_topics=3, embedding_model=mock_embedding_model, random_state=42)
        model.fit(sample_documents)
        
        # Get document info for training docs
        doc_info = model.get_document_info()
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(doc_info, pd.DataFrame)
        assert "Document" in doc_info.columns
        assert "Topic" in doc_info.columns
        assert "Name" in doc_info.columns
        
        # Verify all documents are present
        assert len(doc_info) == len(sample_documents)
        
        # Reset mock to verify transform call for new documents
        mock_embedding_model.embed_documents.reset_mock()
        
        # Get document info for new docs
        new_doc_info = model.get_document_info(sample_documents[:2])
        
        # Verify embedding model was called
        mock_embedding_model.embed_documents.assert_called_once_with(sample_documents[:2])
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(new_doc_info, pd.DataFrame)
        assert len(new_doc_info) == 2
    
    def test_empty_documents(self):
        """Test handling of empty document lists."""
        model = SimpleTopicModel(num_topics=3, random_state=42)
        
        # Fit with empty document list should not raise errors
        model.fit([])
        assert model.is_fitted is False
        
        # Transform with empty document list should return empty result
        result = model.transform([])
        assert result.shape == (0, 3)


class TestTFIDFTopicModel:
    """Test the TFIDFTopicModel class."""
    
    def test_init(self):
        """Test initialization of TFIDFTopicModel."""
        model = TFIDFTopicModel(num_topics=5, max_features=500, random_state=42)
        
        assert model.num_topics == 5
        assert model.max_features == 500
        assert model.random_state == 42
        assert model.is_fitted is False
        assert model.topics == {}
        assert model.topic_words == {}
        assert model.topic_sizes == {}
    
    def test_fit(self, sample_documents):
        """Test fitting the TFIDFTopicModel."""
        model = TFIDFTopicModel(num_topics=3, random_state=42)
        fitted_model = model.fit(sample_documents)
        
        # Verify the model is fitted
        assert fitted_model is model
        assert model.is_fitted is True
        
        # Verify topics were created
        assert len(model.topics) == 3
        assert len(model.topic_words) == 3
        assert len(model.topic_sizes) == 3
        
        # Verify clusters were assigned
        assert hasattr(model, 'clusters')
        assert len(model.clusters) == len(sample_documents)
    
    def test_transform(self, sample_documents):
        """Test transforming documents with TFIDFTopicModel."""
        model = TFIDFTopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Transform new documents
        result = model.transform(sample_documents[:2])
        
        # Verify result shape
        assert result.shape == (2, 3)
        
        # Each document should be assigned to exactly one topic
        assert np.sum(result, axis=1).tolist() == [1.0, 1.0]
    
    def test_get_topic_info(self, sample_documents):
        """Test getting topic info from TFIDFTopicModel."""
        model = TFIDFTopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(topic_info, pd.DataFrame)
        assert "Topic" in topic_info.columns
        assert "Name" in topic_info.columns
        assert "Size" in topic_info.columns
        assert "Words" in topic_info.columns
        
        # Verify all topics are present
        assert len(topic_info) == 3
        assert set(topic_info["Topic"].tolist()) == {0, 1, 2}


class TestNMFTopicModel:
    """Test the NMFTopicModel class."""
    
    def test_init(self):
        """Test initialization of NMFTopicModel."""
        model = NMFTopicModel(num_topics=5, max_features=500, random_state=42)
        
        assert model.num_topics == 5
        assert model.max_features == 500
        assert model.random_state == 42
        assert model.is_fitted is False
        assert model.topics == {}
        assert model.topic_words == {}
        assert model.topic_sizes == {}
    
    def test_fit(self, sample_documents):
        """Test fitting the NMFTopicModel."""
        model = NMFTopicModel(num_topics=3, random_state=42)
        fitted_model = model.fit(sample_documents)
        
        # Verify the model is fitted
        assert fitted_model is model
        assert model.is_fitted is True
        
        # Verify topics were created
        assert len(model.topics) == 3
        assert len(model.topic_words) == 3
        assert len(model.topic_sizes) == 3
        
        # Verify document-topic matrix was created
        assert hasattr(model, 'doc_topic_matrix')
        assert model.doc_topic_matrix.shape == (len(sample_documents), 3)
    
    def test_transform(self, sample_documents):
        """Test transforming documents with NMFTopicModel."""
        model = NMFTopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Transform new documents
        result = model.transform(sample_documents[:2])
        
        # Verify result shape
        assert result.shape == (2, 3)
    
    def test_get_document_info(self, sample_documents):
        """Test getting document info from NMFTopicModel."""
        model = NMFTopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Get document info for training docs
        doc_info = model.get_document_info()
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(doc_info, pd.DataFrame)
        assert "Document" in doc_info.columns
        assert "Topic" in doc_info.columns
        assert "Name" in doc_info.columns
        assert "Weight" in doc_info.columns
        
        # Verify all documents are present
        assert len(doc_info) == len(sample_documents)


class TestLSATopicModel:
    """Test the LSATopicModel class."""
    
    def test_init(self):
        """Test initialization of LSATopicModel."""
        model = LSATopicModel(num_topics=5, max_features=500, random_state=42)
        
        assert model.num_topics == 5
        assert model.max_features == 500
        assert model.random_state == 42
        assert model.is_fitted is False
        assert model.topics == {}
        assert model.topic_words == {}
        assert model.topic_sizes == {}
    
    def test_fit(self, sample_documents):
        """Test fitting the LSATopicModel."""
        model = LSATopicModel(num_topics=3, random_state=42)
        fitted_model = model.fit(sample_documents)
        
        # Verify the model is fitted
        assert fitted_model is model
        assert model.is_fitted is True
        
        # Verify topics were created
        assert len(model.topics) == 3
        assert len(model.topic_words) == 3
        assert len(model.topic_sizes) == 3
        
        # Verify document-topic matrix was created
        assert hasattr(model, 'doc_topic_matrix')
        assert model.doc_topic_matrix.shape == (len(sample_documents), 3)
    
    def test_transform(self, sample_documents):
        """Test transforming documents with LSATopicModel."""
        model = LSATopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Transform new documents
        result = model.transform(sample_documents[:2])
        
        # Verify result shape
        assert result.shape == (2, 3)
    
    def test_get_document_info(self, sample_documents):
        """Test getting document info from LSATopicModel."""
        model = LSATopicModel(num_topics=3, random_state=42)
        model.fit(sample_documents)
        
        # Get document info for training docs
        doc_info = model.get_document_info()
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(doc_info, pd.DataFrame)
        assert "Document" in doc_info.columns
        assert "Topic" in doc_info.columns
        assert "Name" in doc_info.columns
        assert "Weight" in doc_info.columns
        
        # Verify all documents are present
        assert len(doc_info) == len(sample_documents)
        
        # Get document info for new docs
        new_doc_info = model.get_document_info(sample_documents[:2])
        
        # Verify result is a DataFrame with expected columns
        assert isinstance(new_doc_info, pd.DataFrame)
        assert len(new_doc_info) == 2


class TestUnifiedTopicModelerWithLightweightModels:
    """Test the UnifiedTopicModeler class with lightweight models."""
    
    @pytest.mark.parametrize("method", [
        "simple_kmeans", 
        "tfidf", 
        "nmf", 
        "lsa"
    ])
    def test_init_with_lightweight_models(self, method):
        """Test initializing UnifiedTopicModeler with various lightweight models."""
        modeler = UnifiedTopicModeler(
            method=method,
            num_topics=5,
            random_state=42
        )
        
        assert modeler.method == method
        assert modeler.num_topics == 5
        assert modeler.random_state == 42
        
        # Verify the correct model type was created
        if method == "simple_kmeans":
            assert isinstance(modeler.model, SimpleTopicModel)
        elif method == "tfidf":
            assert isinstance(modeler.model, TFIDFTopicModel)
        elif method == "nmf":
            assert isinstance(modeler.model, NMFTopicModel)
        elif method == "lsa":
            assert isinstance(modeler.model, LSATopicModel)
    
    @pytest.mark.parametrize("method", [
        "simple_kmeans", 
        "tfidf", 
        "nmf", 
        "lsa"
    ])
    def test_fit_with_lightweight_models(self, method, sample_documents, mock_embedding_model):
        """Test fitting UnifiedTopicModeler with various lightweight models."""
        # For SimpleTopicModel, we need to pass an embedding model
        kwargs = {}
        if method == "simple_kmeans":
            kwargs["embedding_model"] = mock_embedding_model
            
        modeler = UnifiedTopicModeler(
            method=method,
            num_topics=3,
            random_state=42,
            **kwargs
        )
        
        # Fit the model
        modeler.fit(sample_documents)
        
        # Verify model is fitted
        assert modeler.model.is_fitted is True
        
        # Get topic info
        topic_info = modeler.get_topic_info()
        
        # Verify topic info
        assert isinstance(topic_info, pd.DataFrame)
        assert len(topic_info) == 3
        
        # Transform documents
        doc_topic = modeler.transform(sample_documents[:2])
        
        # Verify transform result
        assert isinstance(doc_topic, np.ndarray)
        assert doc_topic.shape[0] == 2
        
        # Get document info
        doc_info = modeler.get_document_info(sample_documents[:2])
        
        # Verify document info
        assert isinstance(doc_info, pd.DataFrame)
        assert len(doc_info) == 2