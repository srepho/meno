"""Tests for CPU-only embedding functionality."""

import pytest
import numpy as np
import pandas as pd
from typing import List
import torch

try:
    from meno.modeling.embeddings import DocumentEmbedding
    ACTUAL_IMPORTS = True
except ImportError:
    ACTUAL_IMPORTS = False
    pytest.skip("Skipping CPU embedding tests due to missing dependencies", allow_module_level=True)


@pytest.mark.skipif(not ACTUAL_IMPORTS, reason="Requires actual implementation")
class TestCPUEmbedding:
    """Tests specifically targeting CPU-only functionality for embeddings."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for embedding."""
        return [
            "This is the first document about technology.",
            "The second document covers healthcare topics.",
            "Document three is about sports and fitness.",
            "The fourth document discusses education and learning.",
            "Finally, this document covers politics and government."
        ]

    @pytest.fixture
    def embedding_model_cpu_explicit(self):
        """Create embedding model with CPU explicitly specified."""
        return DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            use_gpu=False,
            batch_size=32
        )
    
    @pytest.fixture
    def embedding_model_cpu_implicit(self):
        """Create embedding model with CPU implicitly selected via use_gpu=False."""
        return DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",
            use_gpu=False,
            batch_size=32
        )

    def test_cpu_explicit_device_setting(self, embedding_model_cpu_explicit):
        """Test that the device is correctly set to CPU when explicitly specified."""
        assert embedding_model_cpu_explicit.device == "cpu"
        assert embedding_model_cpu_explicit.model.device == "cpu"
    
    def test_cpu_implicit_device_selection(self, embedding_model_cpu_implicit):
        """Test that device defaults to CPU when use_gpu=False."""
        assert embedding_model_cpu_implicit.device == "cpu"
        assert embedding_model_cpu_implicit.model.device == "cpu"
    
    def test_embed_with_pandas_series(self, embedding_model_cpu_explicit, sample_texts):
        """Test embedding a pandas Series on CPU."""
        # Convert list to pandas Series
        text_series = pd.Series(sample_texts)
        
        # Generate embeddings
        embeddings = embedding_model_cpu_explicit.embed_documents(text_series)
        
        # Check shape and properties
        assert embeddings.shape == (len(sample_texts), 384)
        assert isinstance(embeddings, np.ndarray)
        
        # Check that embeddings are normalized (unit vectors)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_batch_size_effect(self, sample_texts):
        """Test different batch sizes don't affect embedding results on CPU."""
        # Create models with different batch sizes
        model_small_batch = DocumentEmbedding(
            model_name="all-MiniLM-L6-v2", 
            device="cpu",
            batch_size=2
        )
        
        model_large_batch = DocumentEmbedding(
            model_name="all-MiniLM-L6-v2", 
            device="cpu",
            batch_size=16
        )
        
        # Generate embeddings with different batch sizes
        embeddings_small_batch = model_small_batch.embed_documents(sample_texts)
        embeddings_large_batch = model_large_batch.embed_documents(sample_texts)
        
        # Results should be identical regardless of batch size
        assert np.allclose(embeddings_small_batch, embeddings_large_batch, atol=1e-5)

    def test_local_model_loading(self, tmp_path, sample_texts):
        """Test loading a model from a local path."""
        # First, create a model and save it locally
        model = DocumentEmbedding(model_name="all-MiniLM-L6-v2", device="cpu")
        local_path = tmp_path / "test_model"
        model.model.save(str(local_path))
        
        # Now load the model from the local path
        local_model = DocumentEmbedding(local_model_path=str(local_path), device="cpu")
        
        # Test that it works
        embeddings = local_model.embed_documents(sample_texts)
        assert embeddings.shape == (len(sample_texts), 384)
        
        # Both models should give the same results
        embeddings_original = model.embed_documents(sample_texts)
        assert np.allclose(embeddings, embeddings_original, atol=1e-5)

    def test_memory_usage(self, embedding_model_cpu_explicit, sample_texts):
        """Test that memory usage on CPU is reasonable."""
        # Get baseline memory usage
        baseline_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Generate embeddings multiple times
        for _ in range(5):
            embeddings = embedding_model_cpu_explicit.embed_documents(sample_texts)
            assert embeddings.shape == (len(sample_texts), 384)
        
        # Memory usage should not increase on CPU since we're not using CUDA
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            assert current_memory == baseline_memory  # No GPU memory should be used