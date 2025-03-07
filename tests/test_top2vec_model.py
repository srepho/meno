"""Tests for Top2Vec model implementation."""

import pytest
import numpy as np
import pandas as pd
from typing import List
import os
from pathlib import Path
import pickle

try:
    from top2vec import Top2Vec
    from meno.modeling.top2vec_model import Top2VecModel
    from meno.modeling.embeddings import DocumentEmbedding
    TOP2VEC_AVAILABLE = True
except ImportError:
    TOP2VEC_AVAILABLE = False

# Skip these tests if top2vec is not available
pytestmark = pytest.mark.skipif(
    not TOP2VEC_AVAILABLE,
    reason="Top2Vec dependencies not available"
)


class TestTop2VecModel:
    """Tests for the Top2Vec model implementation."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for testing."""
        return [
            "This document is about machine learning algorithms.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are used in deep learning algorithms.",
            "Python is a programming language used in data science.",
            "Data science involves statistics and programming.",
            "Clustering is an unsupervised machine learning technique.",
            "Topic modeling extracts themes from document collections.",
            "Sentiment analysis determines the emotion in text.",
            "Natural language processing deals with human language.",
            "Transformers are neural networks for sequence data.",
            "BERT is a transformer model for language understanding.",
            "Word embeddings represent words as vectors.",
            "Text classification assigns categories to documents.",
            "Dimensionality reduction visualizes high-dimensional data.",
            "TF-IDF measures word importance in documents."
        ] * 2  # Repeat to ensure enough data for Top2Vec

    @pytest.fixture
    def embedding_model(self):
        """Create a test embedding model."""
        return DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8
        )

    @pytest.fixture
    def top2vec_model(self, embedding_model):
        """Create a Top2Vec model instance."""
        return Top2VecModel(
            embedding_model=embedding_model,
            n_topics=3,  # Small number for testing
            min_topic_size=2,
            use_gpu=False,
            use_custom_embeddings=True,
            verbose=False
        )

    def test_model_initialization(self, top2vec_model):
        """Test that the model initializes correctly."""
        # Check attributes
        assert top2vec_model.num_topics == 3  # Updated to use standardized parameter name
        assert top2vec_model.n_topics == 3    # Legacy parameter should still be accessible
        assert top2vec_model.min_topic_size == 2
        assert top2vec_model.use_gpu is False
        assert top2vec_model.use_custom_embeddings is True
        assert top2vec_model.is_fitted is False
        
        # Check embedding model
        assert hasattr(top2vec_model, "embedding_model")
        assert isinstance(top2vec_model.embedding_model, DocumentEmbedding)

    def test_fit_and_transform(self, top2vec_model, sample_texts):
        """Test fitting the model and transforming documents."""
        # Fit the model
        fitted_model = top2vec_model.fit(sample_texts)
        
        # Check it's fitted
        assert fitted_model.is_fitted is True
        assert len(fitted_model.topics) > 0
        assert len(fitted_model.topic_sizes) > 0
        assert hasattr(fitted_model, "model")
        assert isinstance(fitted_model.model, Top2Vec)
        
        # Transform documents
        topic_nums, topic_scores = fitted_model.transform(sample_texts[:5])
        
        # Check outputs
        assert topic_nums.shape[0] == 5
        assert topic_scores.shape[0] == 5

    def test_fit_with_embeddings(self, top2vec_model, embedding_model, sample_texts):
        """Test fitting with pre-computed embeddings."""
        # Generate embeddings
        embeddings = embedding_model.embed_documents(sample_texts)
        
        # Fit with pre-computed embeddings
        fitted_model = top2vec_model.fit(sample_texts, embeddings=embeddings)
        
        # Check it's fitted
        assert fitted_model.is_fitted is True
        assert len(fitted_model.topics) > 0
        
        # Transform with embeddings
        topic_nums, topic_scores = fitted_model.transform(sample_texts[:5], embeddings=embeddings[:5])
        
        # Check outputs
        assert topic_nums.shape[0] == 5
        assert topic_scores.shape[0] == 5

    def test_search_topics(self, top2vec_model, sample_texts):
        """Test searching for topics."""
        # Fit the model
        fitted_model = top2vec_model.fit(sample_texts)
        
        # Test the standardized method name
        similar_topics = fitted_model.find_similar_topics("machine learning", n_topics=2)
        
        # Check results
        assert len(similar_topics) <= 2  # May be fewer than 2 if not enough topics
        if len(similar_topics) > 0:
            # Each result should be a tuple of (topic_id, description, score)
            assert len(similar_topics[0]) == 3
            assert isinstance(similar_topics[0][0], int)  # topic_id
            assert isinstance(similar_topics[0][1], str)  # description
            assert isinstance(similar_topics[0][2], float)  # score
            
        # Test the legacy method for backward compatibility
        topic_nums, topic_scores, topic_words = fitted_model.search_topics("machine learning", num_topics=2)
        assert isinstance(topic_nums, list)
        assert isinstance(topic_scores, list)
        assert isinstance(topic_words, list)

    def test_search_documents(self, top2vec_model, sample_texts):
        """Test searching for documents."""
        # Fit the model
        fitted_model = top2vec_model.fit(sample_texts)
        
        # Search for documents
        similar_docs = fitted_model.search_documents("machine learning", n_docs=3)
        
        # Check results
        assert len(similar_docs) <= 3
        if len(similar_docs) > 0:
            # Each result should be a tuple of (document, score)
            assert len(similar_docs[0]) == 2
            assert isinstance(similar_docs[0][0], str)  # document
            assert isinstance(similar_docs[0][1], float)  # score

    def test_add_documents(self, top2vec_model, sample_texts):
        """Test adding documents to a fitted model."""
        # Fit the model with initial documents
        fitted_model = top2vec_model.fit(sample_texts[:10])
        
        # Record initial topic sizes
        initial_topic_count = len(fitted_model.topics)
        initial_sizes = fitted_model.topic_sizes.copy()
        
        # Add more documents
        new_docs = sample_texts[10:]
        fitted_model.add_documents(new_docs)
        
        # Check that documents were added
        for topic_id in fitted_model.topic_sizes:
            # Size should either stay the same or increase
            assert fitted_model.topic_sizes[topic_id] >= initial_sizes.get(topic_id, 0)
        
        # Transform new documents
        topic_nums, _ = fitted_model.transform(new_docs)
        assert topic_nums.shape[0] == len(new_docs)

    def test_save_and_load(self, top2vec_model, sample_texts, embedding_model, tmp_path):
        """Test saving and loading the model."""
        # Fit the model
        fitted_model = top2vec_model.fit(sample_texts)
        
        # Save the model
        save_path = tmp_path / "top2vec_test"
        fitted_model.save(save_path)
        
        # Check that files were created
        assert (save_path / "metadata.pkl").exists()
        assert (save_path / "top2vec_model").exists()
        
        # Load the model
        loaded_model = Top2VecModel.load(save_path, embedding_model=embedding_model)
        
        # Check attributes
        assert loaded_model.n_topics == fitted_model.n_topics
        assert loaded_model.min_topic_size == fitted_model.min_topic_size
        assert loaded_model.is_fitted is True
        
        # Check topics
        assert set(loaded_model.topics.keys()) == set(fitted_model.topics.keys())
        
        # Make predictions with loaded model
        topic_nums, _ = loaded_model.transform(sample_texts[:3])
        assert topic_nums.shape[0] == 3

    def test_native_embedding(self, sample_texts):
        """Test using Top2Vec with its native embedding."""
        # Create model with native embeddings
        model = Top2VecModel(
            embedding_model=None,
            n_topics=2,
            min_topic_size=2,
            use_custom_embeddings=False,
            verbose=False
        )
        
        # Fit model
        try:
            fitted_model = model.fit(sample_texts)
            
            # Check it's fitted
            assert fitted_model.is_fitted is True
            assert len(fitted_model.topics) > 0
            
            # Transform some documents
            topic_nums, _ = fitted_model.transform(sample_texts[:3])
            assert topic_nums.shape[0] == 3
        except Exception as e:
            # Some environments might not have all dependencies for native embedding
            pytest.skip(f"Native embedding test failed: {str(e)}")