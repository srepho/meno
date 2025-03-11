"""Test script for running the integrated components example.

This script demonstrates that all the components work together correctly:
1. Creates all four types of lightweight models
2. Runs a simple dataset through each model
3. Generates visualizations to compare them
4. Successfully integrates with all components
"""

import os
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Import directly from our modules to test integration
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


def get_sample_data():
    """Get sample data for testing."""
    # Example data - technology, healthcare, and environment topics
    sample_documents = [
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
    return sample_documents


def create_embedding_model_mock():
    """Create a mock embedding model."""
    class MockEmbeddingModel:
        def embed_documents(self, documents):
            """Generate random embeddings for documents."""
            return np.random.random((len(documents), 384))  # 384 is standard embedding dim
            
    return MockEmbeddingModel()


def test_simple_model():
    """Test SimpleTopicModel."""
    print("\nTesting SimpleTopicModel...")
    documents = get_sample_data()
    
    # Create model with mock embedding model
    model = SimpleTopicModel(num_topics=3, random_state=42)
    model.embedding_model = create_embedding_model_mock()
    
    # Fit model
    model.fit(documents)
    
    # Test model functionality
    assert model.is_fitted, "Model should be fitted"
    
    # Get topic info
    topic_info = model.get_topic_info()
    print(f"Found {len(topic_info)} topics:")
    for _, row in topic_info.iterrows():
        print(f"  Topic {row['Topic']}: {row['Name']}")
    
    # Test document assignments
    doc_info = model.get_document_info()
    assert len(doc_info) == len(documents), "Document assignments don't match input size"
    
    # Test topic visualization
    fig = model.visualize_topics()
    assert fig is not None, "Visualization should be created"
    
    # Test transform
    new_docs = ["AI is changing many industries."]
    result = model.transform(new_docs)
    assert len(result) == 2, "Transform should return a tuple of (assignments, doc_topic_matrix)"
    
    return model


def test_tfidf_model():
    """Test TFIDFTopicModel."""
    print("\nTesting TFIDFTopicModel...")
    documents = get_sample_data()
    
    # Create model
    model = TFIDFTopicModel(num_topics=3, random_state=42)
    
    # Fit model
    model.fit(documents)
    
    # Test model functionality
    assert model.is_fitted, "Model should be fitted"
    
    # Get topic info
    topic_info = model.get_topic_info()
    print(f"Found {len(topic_info)} topics:")
    for _, row in topic_info.iterrows():
        print(f"  Topic {row['Topic']}: {row['Name']}")
    
    # Test document assignments
    doc_info = model.get_document_info()
    assert len(doc_info) == len(documents), "Document assignments don't match input size"
    
    # Test topic visualization
    fig = model.visualize_topics()
    assert fig is not None, "Visualization should be created"
    
    # Test transform
    new_docs = ["AI is changing many industries."]
    result = model.transform(new_docs)
    assert len(result) == 2, "Transform should return a tuple of (assignments, doc_topic_matrix)"
    
    return model


def test_nmf_model():
    """Test NMFTopicModel."""
    print("\nTesting NMFTopicModel...")
    documents = get_sample_data()
    
    # Create model
    model = NMFTopicModel(num_topics=3, random_state=42)
    
    # Fit model
    model.fit(documents)
    
    # Test model functionality
    assert model.is_fitted, "Model should be fitted"
    
    # Get topic info
    topic_info = model.get_topic_info()
    print(f"Found {len(topic_info)} topics:")
    for _, row in topic_info.iterrows():
        print(f"  Topic {row['Topic']}: {row['Name']}")
    
    # Test document assignments
    doc_info = model.get_document_info()
    assert len(doc_info) == len(documents), "Document assignments don't match input size"
    
    # Test topic visualization
    fig = model.visualize_topics()
    assert fig is not None, "Visualization should be created"
    
    # Test transform
    new_docs = ["AI is changing many industries."]
    result = model.transform(new_docs)
    assert result.shape == (1, 3), "Transform should return doc_topic_matrix of shape (n_docs, n_topics)"
    
    return model


def test_lsa_model():
    """Test LSATopicModel."""
    print("\nTesting LSATopicModel...")
    documents = get_sample_data()
    
    # Create model
    model = LSATopicModel(num_topics=3, random_state=42)
    
    # Fit model
    model.fit(documents)
    
    # Test model functionality
    assert model.is_fitted, "Model should be fitted"
    
    # Get topic info
    topic_info = model.get_topic_info()
    print(f"Found {len(topic_info)} topics:")
    for _, row in topic_info.iterrows():
        print(f"  Topic {row['Topic']}: {row['Name']}")
    
    # Test document assignments
    doc_info = model.get_document_info()
    assert len(doc_info) == len(documents), "Document assignments don't match input size"
    
    # Test topic visualization
    fig = model.visualize_topics()
    assert fig is not None, "Visualization should be created"
    
    # Test transform
    new_docs = ["AI is changing many industries."]
    result = model.transform(new_docs)
    assert result.shape == (1, 3), "Transform should return doc_topic_matrix of shape (n_docs, n_topics)"
    
    return model


def test_visualizations_integration(models):
    """Test integration with visualizations module."""
    print("\nTesting visualization integration...")
    documents = get_sample_data()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        
        # Test model comparison visualization
        print("  Creating model comparison visualization...")
        model_comparison_fig = plot_model_comparison(
            document_lists=[documents] * len(models),
            model_names=list(models.keys()),
            models=list(models.values())
        )
        assert model_comparison_fig is not None
        
        # Test topic landscape for each model
        print("  Creating topic landscape visualizations...")
        for name, model in models.items():
            fig = plot_topic_landscape(
                model=model,
                documents=documents,
                title=f"{name} Topic Landscape"
            )
            assert fig is not None
        
        # Test multi-topic heatmap with two models
        print("  Creating topic comparison heatmap...")
        model_names = list(models.keys())[:2]
        model_list = [models[name] for name in model_names]
        heatmap_fig = plot_multi_topic_heatmap(
            models=model_list,
            model_names=model_names,
            document_lists=[documents] * 2
        )
        assert heatmap_fig is not None
        
        # Test document analysis
        print("  Creating document analysis visualizations...")
        for name, model in models.items():
            fig = plot_comparative_document_analysis(
                model=model,
                documents=documents,
                title=f"{name} Document Analysis"
            )
            assert fig is not None
    
    print("  All visualizations created successfully")
    return True


def test_model_serialization(models):
    """Test model serialization and deserialization."""
    print("\nTesting model serialization...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, model in models.items():
            # Create save path
            save_path = Path(tmp_dir) / name
            
            # Save model
            print(f"  Saving {name} model...")
            model.save(save_path)
            
            # Verify files were created
            assert (save_path / "model_data.pkl").exists()
            
            # Load model
            print(f"  Loading {name} model...")
            if name == "simple":
                loaded_model = SimpleTopicModel.load(save_path)
            elif name == "tfidf":
                loaded_model = TFIDFTopicModel.load(save_path)
            elif name == "nmf":
                loaded_model = NMFTopicModel.load(save_path)
            elif name == "lsa":
                loaded_model = LSATopicModel.load(save_path)
            
            # Verify model was loaded correctly
            assert loaded_model.is_fitted
            assert loaded_model.num_topics == model.num_topics
            assert loaded_model.topics == model.topics
            print(f"  {name} model serialization successful")
    
    print("  All models serialized and deserialized successfully")
    return True


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("INTEGRATED COMPONENTS TEST")
    print("=" * 80)
    
    # Test all models
    simple_model = test_simple_model()
    tfidf_model = test_tfidf_model()
    nmf_model = test_nmf_model()
    lsa_model = test_lsa_model()
    
    # Collect all models
    models = {
        "simple": simple_model,
        "tfidf": tfidf_model,
        "nmf": nmf_model, 
        "lsa": lsa_model
    }
    
    # Test visualizations
    test_visualizations_integration(models)
    
    # Test model serialization
    test_model_serialization(models)
    
    print("\nAll integration tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())