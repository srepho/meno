"""Functional tests for the embedding model."""

import pytest
import numpy as np
from typing import List

try:
    from meno.modeling.embeddings import DocumentEmbedding
    ACTUAL_IMPORTS = True
except ImportError:
    ACTUAL_IMPORTS = False
    pytest.skip("Skipping functional embedding tests due to missing dependencies", allow_module_level=True)


@pytest.mark.skipif(not ACTUAL_IMPORTS, reason="Requires actual implementation")
class TestEmbeddingFunctional:
    """Functional tests for the embedding model using a real model."""

    @pytest.fixture
    def embedding_model(self):
        """Create a real embedding model with a small, fast model."""
        return DocumentEmbedding(
            # Use a small model for faster tests
            model_name="all-MiniLM-L6-v2", 
            batch_size=32
        )
    
    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for embedding."""
        return [
            "This is a document about technology and computers.",
            "Healthcare and medicine are important for public health.",
            "Sports and exercise are good for physical health.",
            "Education and learning are lifelong pursuits.",
            "Politics and government policies affect many aspects of life."
        ]
    
    def test_embedding_model_dimensions(self, embedding_model, sample_texts):
        """Test that the embedding model produces the expected dimensions."""
        # Embed documents
        embeddings = embedding_model.embed_documents(sample_texts)
        
        # Check dimensions - MiniLM produces 384-dim embeddings
        assert embeddings.shape == (len(sample_texts), 384)
        
        # Check that embeddings are normalized (unit vectors)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_embedding_similarity(self, embedding_model, sample_texts):
        """Test that semantically similar texts have higher cosine similarity."""
        # Embed documents
        embeddings = embedding_model.embed_documents(sample_texts)
        
        # Calculate cosine similarities between all pairs
        # (embeddings are normalized, so dot product equals cosine similarity)
        similarities = np.dot(embeddings, embeddings.T)
        
        # Technology text should be more similar to education than to sports
        tech_idx, edu_idx, sports_idx = 0, 3, 2
        
        tech_edu_sim = similarities[tech_idx, edu_idx]
        tech_sports_sim = similarities[tech_idx, sports_idx]
        
        assert tech_edu_sim > tech_sports_sim
        
        # Healthcare should be more similar to sports than to politics
        health_idx, politics_idx = 1, 4
        
        health_sports_sim = similarities[health_idx, sports_idx]
        health_politics_sim = similarities[health_idx, politics_idx]
        
        assert health_sports_sim > health_politics_sim
    
    def test_topic_embedding(self, embedding_model):
        """Test embedding topics with descriptions."""
        topics = ["Technology", "Healthcare", "Sports"]
        descriptions = [
            "Computers, software, hardware, and IT",
            "Medicine, doctors, hospitals, and patient care",
            "Physical activities, games, and competitions"
        ]
        
        # Embed topics
        topic_embeddings = embedding_model.embed_topics(topics, descriptions)
        
        # Check dimensions
        assert topic_embeddings.shape == (len(topics), 384)
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(topic_embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        
        # Embed just topics without descriptions for comparison
        topics_only_embeddings = embedding_model.embed_topics(topics)
        
        # Topic+description embeddings should be different from topic-only embeddings
        for i in range(len(topics)):
            similarity = np.dot(topic_embeddings[i], topics_only_embeddings[i])
            assert similarity < 0.99  # They should be similar but not identical
    
    def test_batch_consistency(self, embedding_model):
        """Test that batch processing produces the same results as individual processing."""
        # Create larger batch of texts
        texts = [
            "This is the first document.",
            "Here is another document.",
            "This is the third test document.",
            "Finally, this is the last document."
        ]
        
        # Embed as batch
        batch_embeddings = embedding_model.embed_documents(texts)
        
        # Embed individually
        individual_embeddings = np.array([
            embedding_model.embed_documents([text])[0]
            for text in texts
        ])
        
        # Results should be identical
        assert np.allclose(batch_embeddings, individual_embeddings, atol=1e-5)