"""Functional integration tests for the Meno package."""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

try:
    from meno.meno import MenoTopicModeler
    ACTUAL_IMPORTS = True
except ImportError:
    ACTUAL_IMPORTS = False
    pytest.skip("Skipping functional integration tests due to missing dependencies", allow_module_level=True)


@pytest.mark.skipif(not ACTUAL_IMPORTS, reason="Requires actual implementation")
class TestMenoIntegrationFunctional:
    """End-to-end functional tests for the Meno topic modeler."""

    @pytest.fixture
    def sample_documents(self) -> List[str]:
        """Create sample documents for testing."""
        return [
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

    @pytest.fixture
    def sample_config_override(self) -> Dict:
        """Configuration overrides for faster testing."""
        return {
            "modeling": {
                "embeddings": {
                    # Use a smaller, faster model
                    "model_name": "all-MiniLM-L6-v2", 
                    "batch_size": 32
                },
                "clustering": {
                    "algorithm": "kmeans",
                    "n_clusters": 3  # We expect 3 main topics: AI/ML, Healthcare, Environment
                }
            },
            "preprocessing": {
                "normalization": {
                    # Disable lemmatization for faster tests
                    "lemmatize": False
                }
            }
        }

    def test_end_to_end_unsupervised(self, sample_documents, sample_config_override):
        """Test the complete unsupervised topic modeling workflow."""
        # Create topic modeler
        modeler = MenoTopicModeler(config_overrides=sample_config_override)
        
        # Preprocess documents
        processed_df = modeler.preprocess(sample_documents)
        
        # Check preprocessing results
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_documents)
        assert "text" in processed_df.columns
        assert "processed_text" in processed_df.columns
        assert "doc_id" in processed_df.columns
        
        # Generate embeddings
        embeddings = modeler.embed_documents()
        
        # Check embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(sample_documents), 384)  # MiniLM produces 384-dim vectors
        
        # Discover topics
        result_df = modeler.discover_topics(method="embedding_cluster")
        
        # Check topic assignments
        assert isinstance(result_df, pd.DataFrame)
        assert "topic" in result_df.columns
        assert result_df["topic"].nunique() <= 3  # Should have at most 3 topics (k-means)
        
        # Create visualization (just test that it doesn't error)
        fig = modeler.visualize_embeddings(return_figure=True)
        assert fig is not None
        
        # Create topic distribution visualization
        fig2 = modeler.visualize_topic_distribution(return_figure=True)
        assert fig2 is not None
        
        # Test report generation
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.html"
            output_path = modeler.generate_report(output_path=str(report_path))
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Test exporting results
            export_dir = Path(temp_dir) / "exports"
            exports = modeler.export_results(
                output_path=str(export_dir),
                formats=["csv", "json"],
                include_embeddings=False
            )
            
            assert len(exports) == 2
            assert os.path.exists(exports["csv"])
            assert os.path.exists(exports["json"])
            
    def test_supervised_topic_matching(self, sample_documents, sample_config_override):
        """Test supervised topic matching workflow."""
        # Create topic modeler
        modeler = MenoTopicModeler(config_overrides=sample_config_override)
        
        # Preprocess documents
        processed_df = modeler.preprocess(sample_documents)
        
        # Generate embeddings
        _ = modeler.embed_documents()
        
        # Define topics
        topics = ["AI & Machine Learning", "Healthcare Technology", "Environmental Sustainability"]
        descriptions = [
            "Artificial intelligence, machine learning, neural networks, and related computer science topics.",
            "Medical technology, healthcare systems, electronic health records, and patient care.",
            "Climate change, renewable energy, conservation, and sustainable development."
        ]
        
        # Match documents to topics
        result_df = modeler.match_topics(topics, descriptions)
        
        # Check results
        assert isinstance(result_df, pd.DataFrame)
        assert "topic" in result_df.columns
        # Check that all assigned topics (excluding "Unknown") are in our defined topics list
        assigned_topics = set(topic for topic in result_df["topic"].unique() if topic != "Unknown")
        assert assigned_topics.issubset(set(topics))
        
        # For the 1.0.0 release, simply verify that topic assignment works and returns expected structure
        # The actual topic assignment accuracy will be tested in more detail in future releases
        
