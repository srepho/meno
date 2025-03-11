"""Tests for the StreamingProcessor class for large-scale data processing."""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from meno.modeling.streaming_processor import StreamingProcessor
from meno.modeling.embeddings import DocumentEmbedding

# Skip tests if required dependencies are not available
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "text": [
            "This is a document about artificial intelligence and machine learning.",
            "Natural language processing is a subfield of artificial intelligence.",
            "Deep learning is a type of machine learning based on neural networks.",
            "Supervised learning requires labeled training data.",
            "Unsupervised learning finds patterns without labeled data.",
            "Reinforcement learning involves agents taking actions to maximize rewards.",
            "Transfer learning uses knowledge from one task to help with another.",
            "Computer vision is used for image recognition and object detection.",
            "Recurrent neural networks are good for sequential data like text.",
            "Generative adversarial networks can create realistic synthetic data."
        ],
        "id": list(range(10)),
        "category": ["AI", "NLP", "ML", "ML", "ML", "RL", "ML", "CV", "NLP", "GAN"]
    })


@pytest.fixture
def sample_csv_file(temp_dir, sample_data):
    """Create a sample CSV file for testing."""
    file_path = temp_dir / "sample_data.csv"
    sample_data.to_csv(file_path, index=False)
    return file_path


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
def test_stream_from_file(temp_dir, sample_csv_file):
    """Test streaming documents from a file."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Stream documents from CSV file
    doc_batches = []
    id_batches = []
    
    for docs, ids in processor.stream_from_file(
        file_path=sample_csv_file,
        text_column="text",
        id_column="id"
    ):
        doc_batches.append(docs)
        id_batches.append(ids)
    
    # Check number of batches - ceiling division of 10/3
    expected_batches = (10 + 3 - 1) // 3
    assert len(doc_batches) == expected_batches
    
    # Check all documents were processed
    all_docs = [doc for batch in doc_batches for doc in batch]
    all_ids = [id for batch in id_batches for id in batch]
    
    assert len(all_docs) == 10
    assert set(all_ids) == set(range(10))


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
def test_stream_from_file_with_filter(temp_dir, sample_csv_file):
    """Test streaming documents from a file with filtering."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Stream only ML documents
    doc_batches = []
    id_batches = []
    
    for docs, ids in processor.stream_from_file(
        file_path=sample_csv_file,
        text_column="text",
        id_column="id",
        filter_condition="col('category') == 'ML'"
    ):
        doc_batches.append(docs)
        id_batches.append(ids)
    
    # Check all documents were processed
    all_docs = [doc for batch in doc_batches for doc in batch]
    
    # Load original data to verify filter
    original_data = pd.read_csv(sample_csv_file)
    expected_count = len(original_data[original_data["category"] == "ML"])
    
    assert len(all_docs) == expected_count


def test_create_embeddings_stream(temp_dir, sample_data):
    """Test creating embeddings from a stream of documents."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Create a document stream
    doc_stream = [
        (sample_data["text"][:3].tolist(), sample_data["id"][:3].tolist()),
        (sample_data["text"][3:6].tolist(), sample_data["id"][3:6].tolist()),
        (sample_data["text"][6:9].tolist(), sample_data["id"][6:9].tolist()),
        (sample_data["text"][9:].tolist(), sample_data["id"][9:].tolist()),
    ]
    
    # Process document stream
    embedding_batches = []
    id_batches = []
    
    for embeddings, ids in processor.create_embeddings_stream(
        documents_stream=iter(doc_stream),
        save_embeddings=False
    ):
        embedding_batches.append(embeddings)
        id_batches.append(ids)
    
    # Check number of batches
    assert len(embedding_batches) == len(doc_stream)
    
    # Check embedding dimensions
    model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    expected_dim = model.embedding_dimension
    
    for batch in embedding_batches:
        assert batch.shape[1] == expected_dim


def test_create_embeddings_stream_with_saving(temp_dir, sample_data):
    """Test creating and saving embeddings from a stream of documents."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Create a document stream
    doc_stream = [
        (sample_data["text"][:5].tolist(), sample_data["id"][:5].tolist()),
        (sample_data["text"][5:].tolist(), sample_data["id"][5:].tolist()),
    ]
    
    # Define embedding file
    embedding_file = temp_dir / "test_embeddings.npy"
    
    # Process document stream with saving
    embedding_batches = []
    for embeddings, ids in processor.create_embeddings_stream(
        documents_stream=iter(doc_stream),
        save_embeddings=True,
        embedding_file=embedding_file
    ):
        embedding_batches.append(embeddings)
    
    # Check that files were created
    assert embedding_file.exists()
    assert (embedding_file.parent / f"{embedding_file.stem}_ids.json").exists()
    
    # Load saved embeddings and check dimensions
    saved_embeddings = np.load(embedding_file)
    with open(embedding_file.parent / f"{embedding_file.stem}_ids.json", 'r') as f:
        saved_ids = json.load(f)
    
    # Check dimensions
    assert saved_embeddings.shape[0] == len(sample_data)
    assert len(saved_ids) == len(sample_data)


def test_create_embeddings_stream_with_mmap(temp_dir, sample_data):
    """Test creating and saving embeddings with memory-mapped storage."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False,
        use_quantization=True  # Enable quantization for float16 storage
    )
    
    # Create a document stream
    doc_stream = [
        (sample_data["text"][:5].tolist(), sample_data["id"][:5].tolist()),
        (sample_data["text"][5:].tolist(), sample_data["id"][5:].tolist()),
    ]
    
    # Define embedding file
    embedding_file = temp_dir / "test_mmap_embeddings.npy"
    
    # Process document stream with memory-mapped saving
    embedding_batches = []
    for embeddings, ids in processor.create_embeddings_stream(
        documents_stream=iter(doc_stream),
        save_embeddings=True,
        embedding_file=embedding_file,
        use_mmap=True,  # Enable memory-mapped storage
        cache_embeddings=True  # Enable caching
    ):
        embedding_batches.append(embeddings)
    
    # Check that files were created
    assert embedding_file.exists()
    assert (embedding_file.parent / f"{embedding_file.stem}_ids.json").exists()
    assert (embedding_file.parent / f"{embedding_file.stem}_meta.json").exists()
    
    # Load saved embeddings with memory-mapping
    saved_embeddings = np.load(embedding_file, mmap_mode='r')
    with open(embedding_file.parent / f"{embedding_file.stem}_ids.json", 'r') as f:
        saved_ids = json.load(f)
    
    # Check dimensions
    assert saved_embeddings.shape[0] == len(sample_data)
    assert len(saved_ids) == len(sample_data)
    
    # Check quantization if enabled
    if processor.use_quantization:
        assert saved_embeddings.dtype == np.float16
    
    # Load metadata
    with open(embedding_file.parent / f"{embedding_file.stem}_meta.json", 'r') as f:
        metadata = json.load(f)
    
    # Check metadata
    assert "doc_count" in metadata
    assert "embedding_dim" in metadata
    assert "precision" in metadata
    assert "created" in metadata
    assert metadata["doc_count"] == len(sample_data)


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_fit_topic_model_stream(temp_dir, sample_data):
    """Test fitting a topic model using streaming embeddings."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=5,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Generate embeddings for all documents
    model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    all_embeddings = model.embed_documents(sample_data["text"].tolist())
    
    # Create an embeddings stream
    embeddings_stream = [
        (all_embeddings[:5], sample_data["id"][:5].tolist()),
        (all_embeddings[5:], sample_data["id"][5:].tolist()),
    ]
    
    # Fit topic model
    processor.fit_topic_model_stream(
        embeddings_stream=iter(embeddings_stream),
        n_topics=3,  # Small number for testing
        min_topic_size=2,
        method="bertopic"
    )
    
    # Check that the model was fitted
    assert processor.is_fitted
    assert processor.topic_model is not None
    
    # Verify that topic information is available
    topic_info = processor.topic_model.get_topic_info()
    assert len(topic_info) > 0


@pytest.mark.skipif(not POLARS_AVAILABLE or not BERTOPIC_AVAILABLE, 
                  reason="Polars or BERTopic not available")
def test_process_file(temp_dir, sample_csv_file):
    """Test end-to-end file processing."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=5,  # Small batch size for testing
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Process file end-to-end
    results = processor.process_file(
        file_path=sample_csv_file,
        text_column="text",
        id_column="id",
        n_topics=2,  # Small number for testing
        min_topic_size=2,
        topic_method="bertopic",
        save_results=True,
        output_dir=temp_dir
    )
    
    # Check results
    assert "file_processed" in results
    assert "document_count" in results
    assert "batch_count" in results
    assert "timing" in results
    assert "topic_count" in results
    
    # Check output files
    assert Path(results["embedding_file"]).exists()
    assert Path(results["model_path"]).exists()


def test_clean_temp_files(temp_dir, sample_data):
    """Test cleaning temporary files."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=3,
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Create a document stream
    doc_stream = [
        (sample_data["text"][:5].tolist(), sample_data["id"][:5].tolist()),
        (sample_data["text"][5:].tolist(), sample_data["id"][5:].tolist()),
    ]
    
    # Process document stream with saving
    embedding_file = temp_dir / "test_embeddings.npy"
    list(processor.create_embeddings_stream(
        documents_stream=iter(doc_stream),
        save_embeddings=True,
        embedding_file=embedding_file
    ))
    
    # Verify files exist
    assert embedding_file.exists()
    assert (embedding_file.parent / f"{embedding_file.stem}_ids.json").exists()
    
    # Clean temp files
    processor.clean_temp_files()
    
    # Verify files were removed
    assert not embedding_file.exists()
    assert not (embedding_file.parent / f"{embedding_file.stem}_ids.json").exists()


def test_get_stats(temp_dir, sample_data):
    """Test getting processing statistics."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=5,
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Create a document stream
    doc_stream = [
        (sample_data["text"][:5].tolist(), sample_data["id"][:5].tolist()),
        (sample_data["text"][5:].tolist(), sample_data["id"][5:].tolist()),
    ]
    
    # Process documents
    list(processor.create_embeddings_stream(
        documents_stream=iter(doc_stream),
        save_embeddings=False
    ))
    
    # Get stats
    stats = processor.get_stats()
    
    # Check stats
    assert "document_count" in stats
    assert "batch_count" in stats
    assert "timing" in stats
    assert "embedding_dimension" in stats
    assert "model_fitted" in stats
    
    # Check values
    assert stats["document_count"] == len(sample_data)
    assert stats["batch_count"] == 2
    assert not stats["model_fitted"]  # Model not fitted yet


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_apply_to_dataframe(temp_dir, sample_data):
    """Test applying the processor to a DataFrame."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        topic_model="bertopic",
        batch_size=5,
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Generate embeddings for all documents
    model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    all_embeddings = model.embed_documents(sample_data["text"].tolist())
    
    # Create an embeddings stream
    embeddings_stream = [
        (all_embeddings[:5], sample_data["id"][:5].tolist()),
        (all_embeddings[5:], sample_data["id"][5:].tolist()),
    ]
    
    # Fit topic model
    processor.fit_topic_model_stream(
        embeddings_stream=iter(embeddings_stream),
        n_topics=3,
        min_topic_size=2,
        method="bertopic"
    )
    
    # Apply to DataFrame
    result_df = processor.apply_to_dataframe(
        df=sample_data,
        text_column="text",
        output_column="topic",
        output_probs_column="topic_prob"
    )
    
    # Check result
    assert "topic" in result_df.columns
    assert "topic_prob" in result_df.columns
    assert len(result_df) == len(sample_data)
    
    # Check that topics were assigned
    assert result_df["topic"].notna().all()


def test_generate_embeddings_file(temp_dir, sample_csv_file):
    """Test generating embeddings file from a data file."""
    processor = StreamingProcessor(
        embedding_model="all-MiniLM-L6-v2",
        batch_size=5,
        temp_dir=temp_dir,
        verbose=False
    )
    
    # Generate embeddings file
    output_file = temp_dir / "generated_embeddings.npy"
    result_path = processor.generate_embeddings_file(
        file_path=sample_csv_file,
        text_column="text",
        id_column="id",
        output_file=output_file
    )
    
    # Check result
    assert result_path == output_file
    assert output_file.exists()
    assert (output_file.parent / f"{output_file.stem}_ids.json").exists()
    
    # Load embeddings
    embeddings = np.load(output_file)
    with open(output_file.parent / f"{output_file.stem}_ids.json", 'r') as f:
        ids = json.load(f)
    
    # Verify dimensions
    df = pd.read_csv(sample_csv_file)
    assert embeddings.shape[0] == len(df)
    assert len(ids) == len(df)