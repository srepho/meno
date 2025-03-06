"""Tests for BERTopic model implementation."""

import pytest
import numpy as np
import pandas as pd
from typing import List
import os
from pathlib import Path
import tempfile

try:
    from bertopic import BERTopic
    from meno.modeling.bertopic_model import BERTopicModel
    from meno.modeling.embeddings import DocumentEmbedding
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Skip these tests if bertopic is not available
pytestmark = pytest.mark.skipif(
    not BERTOPIC_AVAILABLE,
    reason="BERTopic dependencies not available"
)


class TestBERTopicModel:
    """Tests for the BERTopic model implementation."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for testing."""
        return [
            "This is a document about technology and computers.",
            "Healthcare and medicine are important for public health.",
            "Sports and exercise are good for physical health.",
            "Education and learning are lifelong pursuits.",
            "Politics and government policies affect many aspects of life.",
            "Art and creativity help express human emotions.",
            "Science advances our understanding of the world.",
            "History teaches us about past events and their impact.",
            "Economics studies how resources are allocated.",
            "Philosophy explores fundamental questions about existence."
        ] * 3  # Repeat to have enough documents for BERTopic to find patterns

    @pytest.fixture
    def embedding_model(self):
        """Create a test embedding model."""
        return DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8
        )

    @pytest.fixture
    def bertopic_model(self, embedding_model):
        """Create a BERTopic model instance."""
        return BERTopicModel(
            embedding_model=embedding_model,
            n_topics=3,  # Small number for testing
            min_topic_size=2,
            use_gpu=False,
            n_neighbors=5,
            n_components=2,
            verbose=False
        )

    def test_model_initialization(self, bertopic_model):
        """Test that the model initializes correctly."""
        # Check attributes
        assert bertopic_model.n_topics == 3
        assert bertopic_model.min_topic_size == 2
        assert bertopic_model.use_gpu is False
        assert bertopic_model.is_fitted is False
        
        # Check components
        assert hasattr(bertopic_model, "model")
        assert isinstance(bertopic_model.model, BERTopic)
        assert hasattr(bertopic_model, "embedding_model")
        assert isinstance(bertopic_model.embedding_model, DocumentEmbedding)

    def test_fit_and_transform(self, bertopic_model, sample_texts):
        """Test fitting the model and transforming documents."""
        # Fit the model
        fitted_model = bertopic_model.fit(sample_texts)
        
        # Check it's fitted
        assert fitted_model.is_fitted is True
        assert len(fitted_model.topics) > 0
        assert len(fitted_model.topic_sizes) > 0
        
        # Check topic embeddings
        assert fitted_model.topic_embeddings is not None
        assert fitted_model.topic_embeddings.shape[1] == 384  # MiniLM dimension
        
        # Transform documents
        topics, probs = fitted_model.transform(sample_texts[:5])
        
        # Check outputs
        assert len(topics) == 5
        assert probs.shape[0] == 5

    def test_fit_with_embeddings(self, bertopic_model, embedding_model, sample_texts):
        """Test fitting with pre-computed embeddings."""
        # Generate embeddings
        embeddings = embedding_model.embed_documents(sample_texts)
        
        # Fit with pre-computed embeddings
        fitted_model = bertopic_model.fit(sample_texts, embeddings=embeddings)
        
        # Check it's fitted
        assert fitted_model.is_fitted is True
        assert len(fitted_model.topics) > 0
        
        # Transform with embeddings
        topics, probs = fitted_model.transform(sample_texts[:5], embeddings=embeddings[:5])
        
        # Check outputs
        assert len(topics) == 5
        assert probs.shape[0] == 5

    def test_find_similar_topics(self, bertopic_model, sample_texts):
        """Test finding similar topics."""
        # Fit the model
        fitted_model = bertopic_model.fit(sample_texts)
        
        # Find similar topics
        similar_topics = fitted_model.find_similar_topics("technology and computers", n_topics=2)
        
        # Check results
        assert len(similar_topics) <= 2  # May be fewer than 2 if not enough topics
        if len(similar_topics) > 0:
            # Each result should be a tuple of (topic_id, description, score)
            assert len(similar_topics[0]) == 3
            assert isinstance(similar_topics[0][0], int)  # topic_id
            assert isinstance(similar_topics[0][1], str)  # description
            assert isinstance(similar_topics[0][2], float)  # score

    def test_save_and_load(self, bertopic_model, sample_texts, embedding_model, tmp_path):
        """Test saving and loading the model."""
        # Fit the model
        fitted_model = bertopic_model.fit(sample_texts)
        
        # Save the model
        save_path = tmp_path / "bertopic_test"
        fitted_model.save(save_path)
        
        # Check that files were created
        assert (save_path / "metadata.json").exists()
        assert (save_path / "topic_embeddings.npy").exists()
        
        # Load the model
        loaded_model = BERTopicModel.load(save_path, embedding_model=embedding_model)
        
        # Check attributes
        assert loaded_model.n_topics == fitted_model.n_topics
        assert loaded_model.min_topic_size == fitted_model.min_topic_size
        assert loaded_model.is_fitted is True
        
        # Check topics
        assert set(loaded_model.topics.keys()) == set(fitted_model.topics.keys())
        
        # Make predictions with loaded model
        topics, probs = loaded_model.transform(sample_texts[:3])
        assert len(topics) == 3

    def test_visualizations(self, bertopic_model, sample_texts):
        """Test visualization methods."""
        # Fit the model
        fitted_model = bertopic_model.fit(sample_texts)
        
        # Get visualizations
        topics_viz = fitted_model.visualize_topics()
        hierarchy_viz = fitted_model.visualize_hierarchy()
        
        # Check that visualizations were created
        assert topics_viz is not None
        assert hierarchy_viz is not None
        
        # Basic checks on the visualization objects
        assert hasattr(topics_viz, "data")
        assert hasattr(hierarchy_viz, "data")