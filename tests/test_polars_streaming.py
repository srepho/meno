"""Tests for Polars streaming integration with document embeddings."""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from typing import List, Generator

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from meno.modeling.embeddings import DocumentEmbedding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Skip these tests if required dependencies are not available
pytestmark = pytest.mark.skipif(
    not (POLARS_AVAILABLE and EMBEDDINGS_AVAILABLE),
    reason="Polars and embedding dependencies required"
)


class TestPolarsStreaming:
    """Tests for Polars streaming integration with document embeddings."""

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
        ]

    @pytest.fixture
    def csv_file(self, sample_texts, tmp_path):
        """Create a temporary CSV file with sample texts."""
        # Create DataFrame with texts
        df = pd.DataFrame({
            "text": sample_texts,
            "category": [f"Category {i}" for i in range(len(sample_texts))],
            "id": list(range(len(sample_texts)))
        })
        
        # Save to CSV
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)

    @pytest.fixture
    def parquet_file(self, sample_texts, tmp_path):
        """Create a temporary Parquet file with sample texts."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Create DataFrame with texts
        pl_df = pl.DataFrame({
            "text": sample_texts,
            "category": [f"Category {i}" for i in range(len(sample_texts))],
            "id": list(range(len(sample_texts)))
        })
        
        # Save to Parquet
        parquet_path = tmp_path / "test_data.parquet"
        pl_df.write_parquet(parquet_path)
        
        return str(parquet_path)

    @pytest.fixture
    def embedding_model(self):
        """Create a test embedding model."""
        return DocumentEmbedding(
            model_name="all-MiniLM-L6-v2",  # Smaller model for tests
            device="cpu",
            batch_size=2  # Small batch size for testing
        )

    def test_polars_dataframe_input(self, embedding_model, sample_texts):
        """Test embedding a Polars DataFrame."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Create a Polars DataFrame
        pl_df = pl.DataFrame({
            "text": sample_texts[:5],
            "category": [f"Category {i}" for i in range(5)]
        })
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(pl_df, text_column="text")
        
        # Check dimensions
        assert embeddings.shape == (5, 384)  # MiniLM produces 384-dim embeddings
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_polars_series_input(self, embedding_model, sample_texts):
        """Test embedding a Polars Series."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Create a Polars Series
        pl_series = pl.Series("text", sample_texts[:5])
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(pl_series)
        
        # Check dimensions
        assert embeddings.shape == (5, 384)
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_streaming_csv(self, embedding_model, csv_file):
        """Test streaming embeddings from a CSV file."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Stream embeddings
        batches = list(embedding_model.embed_polars_streaming(
            data_path=csv_file,
            text_column="text",
            batch_size=3  # Small batch size for testing
        ))
        
        # Check that we got the right number of batches
        assert len(batches) == 4  # 10 documents with batch_size=3 should give 4 batches
        
        # Check dimensions of each batch
        assert batches[0].shape == (3, 384)
        assert batches[1].shape == (3, 384)
        assert batches[2].shape == (3, 384)
        assert batches[3].shape == (1, 384)  # Last batch has just 1 document
        
        # Concatenate all batches
        all_embeddings = np.vstack(batches)
        
        # Check total dimensions
        assert all_embeddings.shape == (10, 384)
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(all_embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_streaming_parquet(self, embedding_model, parquet_file):
        """Test streaming embeddings from a Parquet file."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Stream embeddings
        batches = list(embedding_model.embed_polars_streaming(
            data_path=parquet_file,
            text_column="text",
            batch_size=4  # Small batch size for testing
        ))
        
        # Check that we got the right number of batches
        assert len(batches) == 3  # 10 documents with batch_size=4 should give 3 batches
        
        # Check dimensions of each batch
        assert batches[0].shape == (4, 384)
        assert batches[1].shape == (4, 384)
        assert batches[2].shape == (2, 384)  # Last batch has 2 documents
        
        # Concatenate all batches
        all_embeddings = np.vstack(batches)
        
        # Check total dimensions
        assert all_embeddings.shape == (10, 384)

    def test_streaming_with_filter(self, embedding_model, parquet_file):
        """Test streaming with a filter condition."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        # Define a filter function that only keeps even IDs
        def filter_even_ids(reader):
            return reader.filter(pl.col("id") % 2 == 0)
        
        # Stream embeddings with filter
        batches = list(embedding_model.embed_polars_streaming(
            data_path=parquet_file,
            text_column="text",
            batch_size=2,
            filter_condition=filter_even_ids
        ))
        
        # Check dimensions - should have 5 documents with even IDs
        all_embeddings = np.vstack(batches)
        assert all_embeddings.shape == (5, 384)

    def test_document_iterator_streaming(self, embedding_model, sample_texts):
        """Test streaming with a document iterator."""
        # Create a simple iterator that yields documents one at a time
        def document_iterator():
            for text in sample_texts:
                yield text
        
        # Stream embeddings
        batches = list(embedding_model.embed_documents_stream(
            documents_iterator=document_iterator(),
            batch_size=3
        ))
        
        # Check that we got the right number of batches
        assert len(batches) == 4  # 10 documents with batch_size=3 should give 4 batches
        
        # Concatenate all batches
        all_embeddings = np.vstack(batches)
        
        # Check total dimensions
        assert all_embeddings.shape == (10, 384)
        
        # Compare with non-streaming version
        direct_embeddings = embedding_model.embed_documents(sample_texts)
        
        # The results should be identical
        assert np.allclose(all_embeddings, direct_embeddings, atol=1e-5)