"""Tests for memory-mapped storage functionality."""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

from meno.modeling.embeddings import DocumentEmbedding
from meno.visualization.umap_viz import UMAPCache, set_umap_cache_dir, get_umap_cache

# Test data
TEST_TEXTS = [
    "This is a test document about technology and computers.",
    "Financial services are important for the economy.",
    "Healthcare should be accessible to everyone.",
    "Education is the key to a better future.",
    "Environmental protection is crucial for our planet.",
    "The weather today is quite nice and sunny.",
    "Sports competitions bring people together.",
    "Music has the power to change your mood.",
    "Food is an important part of every culture.",
    "Travel broadens your horizons and perspective."
]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_embedding_cache_creation():
    """Test that embedding cache is created correctly."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        cache_dir = Path(tmpdirname) / "embedding_cache"
        embedding_model = DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",  # Small model for testing
            cache_dir=str(cache_dir),
            use_mmap=True
        )
        
        # Check that cache directory was created
        assert cache_dir.exists()
        assert cache_dir.is_dir()


def test_embedding_caching(temp_cache_dir):
    """Test that embeddings are properly cached."""
    cache_dir = Path(temp_cache_dir) / "embedding_cache"
    
    # Create embedding model with caching
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",  # Small model for testing
        cache_dir=str(cache_dir),
        use_mmap=True
    )
    
    # Generate embeddings with caching enabled
    embeddings = embedding_model.embed_documents(TEST_TEXTS, cache=True)
    
    # Check that cache files were created
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) > 0, "No cache files were created"
    
    # Get cache info
    cache_info = embedding_model.get_cache_info()
    assert cache_info["entry_count"] > 0
    assert cache_info["total_size_bytes"] > 0
    
    # Generate embeddings again - should use cache
    embeddings_cached = embedding_model.embed_documents(TEST_TEXTS, cache=True)
    
    # Check that embeddings are the same
    np.testing.assert_array_equal(embeddings, embeddings_cached)


def test_float16_precision(temp_cache_dir):
    """Test that float16 precision works correctly."""
    cache_dir = Path(temp_cache_dir) / "embedding_cache"
    
    # Create embedding model with float16 precision
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",
        cache_dir=str(cache_dir),
        use_mmap=True,
        precision="float16"
    )
    
    # Generate embeddings with float16 precision
    embeddings = embedding_model.embed_documents(TEST_TEXTS, cache=True)
    
    # Check that embeddings are float16
    assert embeddings.dtype == np.float16
    
    # Check cache files
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) > 0
    
    # Load cached embeddings and check dtype
    cache_info = embedding_model.get_cache_info()
    for entry in cache_info["entries"]:
        cache_id = entry["cache_id"]
        cached_embeddings = embedding_model.get_cached_embeddings(cache_id)
        assert cached_embeddings.dtype == np.float16


def test_umap_cache(temp_cache_dir):
    """Test UMAP cache functionality."""
    umap_cache_dir = Path(temp_cache_dir) / "umap_cache"
    umap_cache_dir.mkdir(exist_ok=True)
    
    # Set UMAP cache directory
    set_umap_cache_dir(str(umap_cache_dir))
    
    # Get cache instance
    umap_cache = get_umap_cache()
    
    # Create some test embeddings
    embeddings = np.random.rand(10, 20)
    
    # Create UMAP projection manually
    projection = np.random.rand(10, 2)
    
    # Save to cache
    umap_cache.save(
        projection=projection,
        embeddings=embeddings,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine"
    )
    
    # Check that cache files were created
    cache_files = list(umap_cache_dir.glob("*.npy"))
    assert len(cache_files) > 0, "No UMAP cache files were created"
    
    # Retrieve from cache
    cached_projection = umap_cache.get(
        embeddings=embeddings,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine"
    )
    
    # Check that projection is the same
    assert cached_projection is not None
    np.testing.assert_array_equal(projection, cached_projection)


def test_clear_cache(temp_cache_dir):
    """Test clearing the cache."""
    cache_dir = Path(temp_cache_dir) / "embedding_cache"
    
    # Create embedding model with caching
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",
        cache_dir=str(cache_dir),
        use_mmap=True
    )
    
    # Generate embeddings with caching enabled
    embeddings = embedding_model.embed_documents(TEST_TEXTS, cache=True)
    
    # Check that cache files were created
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) > 0
    
    # Clear cache
    embedding_model.clear_cache()
    
    # Check that cache files were removed
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) == 0