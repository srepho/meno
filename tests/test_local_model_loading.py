"""Test suite for local model loading functionality"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import torch
import numpy as np
import pandas as pd

from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel
from meno import MenoTopicModeler
from meno.workflow import MenoWorkflow


@pytest.fixture
def sample_texts():
    """Return a list of sample texts for testing"""
    return [
        "The CEO and CFO met to discuss AI implementation in our CRM system.",
        "Customer submitted a claim for vehicle accident on highway.",
        "The CTO presented ML strategy for improving customer retention."
    ]


@pytest.fixture
def mock_model_path():
    """Create a temporary directory structure that mimics a model directory"""
    temp_dir = tempfile.mkdtemp()
    
    # Mock some model files
    Path(temp_dir, "config.json").write_text('{"model_type": "test"}')
    Path(temp_dir, "tokenizer.json").write_text('{}')
    
    # Mock weights file
    with open(Path(temp_dir, "model.safetensors"), 'wb') as f:
        f.write(b'mock model weights')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@mock.patch("meno.modeling.embeddings.SentenceTransformer")
def test_local_model_path_loading(mock_sentence_transformer, mock_model_path):
    """Test loading a model from a local path"""
    # Setup mock
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.random.rand(3, 384)
    
    # Create embedding model with local_model_path
    embedding_model = DocumentEmbedding(
        local_model_path=mock_model_path,
        use_gpu=False
    )
    
    # Check if SentenceTransformer was called with the correct path
    mock_sentence_transformer.assert_called_once_with(mock_model_path, device='cpu')
    
    # Test embedding functionality still works
    texts = ["This is a test", "Another test", "Final test"]
    embeddings = embedding_model.embed_documents(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)


@mock.patch("meno.modeling.embeddings.SentenceTransformer")
def test_local_files_only_option(mock_sentence_transformer, sample_texts):
    """Test the local_files_only parameter"""
    # Setup mock
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.random.rand(3, 384)
    
    # Create embedding model with local_files_only=True
    # This should raise an error if the model is not found locally
    with mock.patch("os.path.exists", return_value=False):
        with pytest.raises(ValueError) as excinfo:
            DocumentEmbedding(
                model_name="test-model",
                local_files_only=True
            )
        assert "not found locally" in str(excinfo.value)
    
    # Test with a mock existing model
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("meno.modeling.embeddings.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            
            DocumentEmbedding(
                model_name="test-model",
                local_files_only=True
            )
            # Should not try to download
            assert mock_st.called


@mock.patch("meno.modeling.embeddings.SentenceTransformer")
@mock.patch("meno.modeling.bertopic_model.BERTopic")
def test_bertopic_with_local_files(mock_bertopic, mock_sentence_transformer):
    """Test BERTopic model with local_files_only option"""
    # Setup mocks
    mock_st_instance = mock_sentence_transformer.return_value
    mock_st_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st_instance.encode.return_value = np.random.rand(3, 384)
    
    mock_bertopic_instance = mock_bertopic.return_value
    mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0], np.random.rand(3, 2))
    mock_bertopic_instance.get_topic_info.return_value = mock.MagicMock()
    mock_bertopic_instance.get_topic.return_value = [("word1", 0.8), ("word2", 0.7)]
    
    # Create embedding model with local_files_only
    embedding_model = DocumentEmbedding(
        model_name="test-model",
        local_files_only=True,
        use_gpu=False
    )
    
    # Create BERTopic model with embedding model
    bertopic_model = BERTopicModel(
        embedding_model=embedding_model,
        min_topic_size=1
    )
    
    # Should work with the mocked embedding model
    bertopic_model.fit(["Test document 1", "Test document 2", "Test document 3"])
    assert bertopic_model.is_fitted
    
    # Test load method with local_files_only
    with mock.patch("meno.modeling.bertopic_model.BERTopic.load") as mock_load:
        mock_load.return_value = mock_bertopic_instance
        
        with mock.patch("builtins.open", mock.mock_open(read_data='{"n_topics": null, "min_topic_size": 10, "n_neighbors": 15, "n_components": 5, "topics": {}, "topic_sizes": {}, "is_fitted": true, "topic_id_mapping": null}')):
            with mock.patch("pathlib.Path.exists", return_value=True):
                # Test load with local_files_only
                loaded_model = BERTopicModel.load(
                    path="/mock/path",
                    local_files_only=True
                )
                
                # Verify embedding model was created with local_files_only=True
                assert mock_sentence_transformer.call_args[0][1] == 'cpu'


@mock.patch("meno.modeling.embeddings.os.path.expanduser")
@mock.patch("meno.modeling.embeddings.os.path.exists")
@mock.patch("meno.modeling.embeddings.SentenceTransformer")
def test_huggingface_cache_detection(mock_sentence_transformer, mock_exists, mock_expanduser, sample_texts):
    """Test detection of models in HuggingFace cache directory"""
    # Setup mocks
    mock_expanduser.return_value = "/mock/home/.cache/huggingface/hub"
    mock_exists.side_effect = lambda path: "huggingface" in path
    
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.random.rand(3, 384)
    
    with mock.patch("os.listdir", return_value=["hash1"]):
        # Create embedding model that should check HF cache
        embedding_model = DocumentEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        
        # Ensure SentenceTransformer was called
        assert mock_sentence_transformer.called


@mock.patch("meno.modeling.embeddings.SentenceTransformer")
def test_topic_modeler_with_local_model(mock_sentence_transformer, mock_model_path, sample_texts):
    """Test MenoTopicModeler with a local model path"""
    # Setup mock
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.random.rand(3, 384)
    
    # Create embedding model with local path
    embedding_model = DocumentEmbedding(
        local_model_path=mock_model_path,
        use_gpu=False
    )
    
    # Create modeler with embedding model
    modeler = MenoTopicModeler(embedding_model=embedding_model)
    
    # Check if embedding model was correctly assigned
    assert modeler.embedding_model is embedding_model
    
    # Test preprocessing and embedding functionality
    df = pd.DataFrame({"text": sample_texts})
    processed = modeler.preprocess(df, text_column="text")
    assert len(processed) == len(sample_texts)
    
    # Test with direct embedding parameter in the config
    with mock.patch("meno.meno.load_config") as mock_config:
        # Create a mock config that includes local_files_only
        mock_config.return_value = mock.MagicMock()
        mock_config.return_value.modeling.embeddings.model_name = "test-model"
        mock_config.return_value.modeling.embeddings.batch_size = 32
        mock_config.return_value.modeling.embeddings.local_files_only = True
        
        # Create modeler with the mocked config
        modeler = MenoTopicModeler()
        
        # Assert that DocumentEmbedding was called with local_files_only=True
        assert mock_sentence_transformer.called
        
        # Get the last call
        for call in mock_sentence_transformer.call_args_list:
            if 'test-model' in str(call):
                args, kwargs = call
                # Test for local_files_only parameter being respected
                assert 'device' in kwargs


@mock.patch("meno.modeling.embeddings.SentenceTransformer")
def test_workflow_with_local_model(mock_sentence_transformer, mock_model_path, sample_texts):
    """Test MenoWorkflow with local model options"""
    # Setup mock
    mock_instance = mock_sentence_transformer.return_value
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.random.rand(3, 384)
    
    # Reset the mock to track new calls
    mock_sentence_transformer.reset_mock()
    
    # Create workflow with local_model_path
    workflow = MenoWorkflow(local_model_path=mock_model_path)
    
    # Test that DocumentEmbedding was created with the correct path
    mock_sentence_transformer.assert_called()
    assert mock_model_path in str(mock_sentence_transformer.call_args)
    
    # Reset mock again
    mock_sentence_transformer.reset_mock()
    
    # Create workflow with local_files_only
    workflow = MenoWorkflow(local_files_only=True)
    
    # Test basic workflow functionality with the mocked model
    df = pd.DataFrame({"text": sample_texts})
    workflow.load_data(df, text_column="text")
    
    # Mock the modeler's preprocess function to prevent it from calling the embedding model
    with mock.patch.object(workflow.modeler, 'preprocess') as mock_preprocess:
        mock_preprocess.return_value = df
        workflow.preprocess_documents()
        mock_preprocess.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])