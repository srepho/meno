"""Tests for the enhanced LLM topic labeling functionality."""

import pytest
import os
import json
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open

# Skip tests if transformers or openai is not available
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Create a mock OpenAI module if it's not available
if not OPENAI_AVAILABLE:
    # Create mock class
    class MockOpenAI:
        pass
    
    class MockAzureOpenAI(MockOpenAI):
        pass
        
    # Create mock module
    class MockOpenAIModule:
        def __init__(self):
            self.OpenAI = MockOpenAI
            self.AzureOpenAI = MockAzureOpenAI
    
    # Replace the real openai with our mock
    import sys
    sys.modules['openai'] = MockOpenAIModule()
    openai = MockOpenAIModule()

from meno.modeling.llm_topic_labeling import LLMTopicLabeler


@pytest.fixture
def sample_texts():
    """Create sample texts for classification testing."""
    return [
        "The latest technology uses artificial intelligence to improve software development",
        "Patients can now access their medical records online through a secure portal",
        "Stock market volatility has increased due to economic uncertainty",
        "New programming language features improve developer productivity",
        "The hospital implemented a new electronic health record system"
    ]


@pytest.fixture
def similar_texts():
    """Create texts with similar content for deduplication testing."""
    return [
        "Python is a programming language used in data science",
        "Python programming language is popular in data science",  # Similar to first
        "Machine learning algorithms can process large amounts of data",
        "AI systems can learn from large volumes of data",  # Similar to third
        "Neural networks are a type of deep learning algorithm",
        "JavaScript is used for web development",
        "Web developers often use JavaScript for client-side scripting"  # Similar to sixth
    ]


class TestEnhancedLLMTopicLabeling:
    """Test the enhanced features of LLMTopicLabeler."""

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_caching_with_batch_processing(self, mock_openai, sample_texts):
        """Test that caching works correctly with batch processing."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Create a mock response for the first call
        mock_response = """TEXT 1: Technology (confidence: HIGH)
TEXT 2: Healthcare (confidence: MEDIUM)
TEXT 3: Finance (confidence: HIGH)
TEXT 4: Technology (confidence: MEDIUM)
TEXT 5: Healthcare (confidence: HIGH)"""
        
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = mock_response
        mock_client.chat.completions.create.return_value = mock_response_obj
        
        # Create a temporary directory for cache
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create labeler with caching enabled
            labeler = LLMTopicLabeler(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                batch_size=5,
                enable_cache=True,
                cache_dir=tmpdir
            )
            
            # First call - should generate and cache results
            first_results = labeler.classify_texts(sample_texts, progress_bar=False)
            
            # Verify API was called once
            assert mock_client.chat.completions.create.call_count == 1
            
            # Now reset the mock and call again with same texts
            mock_client.chat.completions.create.reset_mock()
            
            # Second call - should use cached results
            second_results = labeler.classify_texts(sample_texts, progress_bar=False)
            
            # Verify API was not called again
            assert mock_client.chat.completions.create.call_count == 0
            
            # Verify results are the same
            assert first_results == second_results
            
            # Check that cache files were created
            cache_files = os.listdir(tmpdir)
            assert len(cache_files) > 0
            
            # Modify one text slightly and verify partial cache hit
            modified_texts = sample_texts.copy()
            modified_texts[2] = "A completely new text that was not cached before"
            
            # Mock response for the partial batch (should only contain the new text)
            modified_response = "TEXT 1: News (confidence: HIGH)"
            modified_response_obj = MagicMock()
            modified_response_obj.choices = [MagicMock()]
            modified_response_obj.choices[0].message.content = modified_response
            mock_client.chat.completions.create.return_value = modified_response_obj
            
            # Reset mock for the third call
            mock_client.chat.completions.create.reset_mock()
            
            # Third call with partial new content
            third_results = labeler.classify_texts(modified_texts, progress_bar=False)
            
            # Verify API was called once for the single new text
            assert mock_client.chat.completions.create.call_count == 1
            
            # Check for preserved results and new result
            assert third_results[0] == first_results[0]  # Cached
            assert third_results[1] == first_results[1]  # Cached
            assert third_results[2] == "News"            # New result
            assert third_results[3] == first_results[3]  # Cached
            assert third_results[4] == first_results[4]  # Cached

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_cache_expiration(self, mock_openai):
        """Test that cached results expire after TTL."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Mock response
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = "TEXT 1: Technology (confidence: HIGH)"
        mock_client.chat.completions.create.return_value = mock_response_obj
        
        # Create a temporary directory for cache
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create labeler with short TTL for testing
            labeler = LLMTopicLabeler(
                model_type="openai",
                enable_cache=True,
                cache_dir=tmpdir,
                cache_ttl=2  # Very short TTL (2 seconds)
            )
            
            # Generate a test text
            text = ["Sample text for classification"]
            
            # First call - should generate and cache
            first_result = labeler.classify_texts(text, progress_bar=False)
            
            # Reset mock
            mock_client.chat.completions.create.reset_mock()
            
            # Second call immediately - should use cache
            second_result = labeler.classify_texts(text, progress_bar=False)
            assert mock_client.chat.completions.create.call_count == 0
            
            # Wait for cache to expire
            time.sleep(3)
            
            # Now mock a different response
            mock_response_obj.choices[0].message.content = "TEXT 1: Science (confidence: MEDIUM)"
            
            # Third call after expiry - should generate new result
            third_result = labeler.classify_texts(text, progress_bar=False)
            
            # Verify API was called again
            assert mock_client.chat.completions.create.call_count == 1
            
            # Results should be different
            assert first_result[0] == "Technology"
            assert third_result[0] == "Science"

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_confidence_scores_tracking(self, mock_openai, sample_texts):
        """Test that confidence scores are tracked correctly."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Create varying confidence scores in the response
        mock_response = """TEXT 1: Technology (confidence: HIGH)
TEXT 2: Healthcare (confidence: MEDIUM)
TEXT 3: Finance (confidence: LOW)
TEXT 4: Technology (confidence: 0.95)
TEXT 5: Healthcare (confidence: 0.42)"""
        
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = mock_response
        mock_client.chat.completions.create.return_value = mock_response_obj
        
        # Create labeler
        labeler = LLMTopicLabeler(
            model_type="openai",
            model_name="gpt-3.5-turbo",
            batch_size=5
        )
        
        # Process texts
        results = labeler.classify_texts(sample_texts, progress_bar=False)
        
        # Check results
        assert len(results) == 5
        
        # Verify confidence scores were parsed correctly
        assert labeler.confidence_scores[0] == 0.9  # HIGH
        assert labeler.confidence_scores[1] == 0.7  # MEDIUM
        assert labeler.confidence_scores[2] == 0.5  # LOW
        assert labeler.confidence_scores[3] == 0.95  # Explicit value
        assert labeler.confidence_scores[4] == 0.42  # Explicit value
        
        # Verify confidence scores can be retrieved
        assert 0 in labeler.confidence_scores
        assert 4 in labeler.confidence_scores
        assert len(labeler.confidence_scores) == 5

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_deduplication_functionality(self, mock_openai, similar_texts):
        """Test the deduplication functionality with similar texts."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Mock a response for non-duplicate texts
        mock_response = """TEXT 1: Python (confidence: HIGH)
TEXT 2: Machine Learning (confidence: HIGH)
TEXT 3: Neural Networks (confidence: HIGH)
TEXT 4: Web Development (confidence: HIGH)"""
        
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = mock_response
        mock_client.chat.completions.create.return_value = mock_response_obj
        
        # Create a labeler with deduplication enabled
        labeler = LLMTopicLabeler(
            model_type="openai",
            deduplicate=True,
            deduplication_threshold=0.7  # Set threshold to detect our similar texts
        )
        
        # Instead of mocking _identify_fuzzy_duplicates, let's test the real implementation
        # Process texts with actual deduplication
        results = labeler.classify_texts(similar_texts, progress_bar=False)
        
        # Verify all 7 texts got results
        assert len(results) == 7
        
        # Check that the API was called with fewer than 7 texts
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        
        # We should see a combined message with only 4 texts (non-duplicates)
        combined_message = call_args["messages"][1]["content"]
        
        # The exact number of prompts might vary based on the implementation
        # but it should be fewer than the total texts due to deduplication
        text_count = combined_message.count("TEXT ")
        assert text_count < len(similar_texts)
        
        # Check that duplicate texts got the same classification
        if "Python" in results[0]:
            assert "Python" in results[1]  # Similar Python texts
        
        if "Web Development" in results[5]:
            assert "Web Development" in results[6]  # Similar web dev texts

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_batch_size_limits(self, mock_openai):
        """Test that batching respects size limits and processes in chunks."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Create a list of 25 texts
        large_text_list = [f"Sample text {i} for classification" for i in range(25)]
        
        # Create response for first batch
        first_batch_response = "\n".join([f"TEXT {i+1}: Category {i+1} (confidence: HIGH)" for i in range(10)])
        first_response_obj = MagicMock()
        first_response_obj.choices = [MagicMock()]
        first_response_obj.choices[0].message.content = first_batch_response
        
        # Create response for second batch
        second_batch_response = "\n".join([f"TEXT {i+1}: Category {i+11} (confidence: MEDIUM)" for i in range(10)])
        second_response_obj = MagicMock()
        second_response_obj.choices = [MagicMock()]
        second_response_obj.choices[0].message.content = second_batch_response
        
        # Create response for third batch
        third_batch_response = "\n".join([f"TEXT {i+1}: Category {i+21} (confidence: LOW)" for i in range(5)])
        third_response_obj = MagicMock()
        third_response_obj.choices = [MagicMock()]
        third_response_obj.choices[0].message.content = third_batch_response
        
        # Setup mock to return different responses for each batch
        mock_client.chat.completions.create.side_effect = [
            first_response_obj,
            second_response_obj,
            third_response_obj
        ]
        
        # Create labeler with batch size 10
        labeler = LLMTopicLabeler(
            model_type="openai",
            batch_size=10,  # Process 10 at a time
            deduplicate=False  # No deduplication for this test
        )
        
        # Process all texts
        results = labeler.classify_texts(large_text_list, progress_bar=False)
        
        # Verify all 25 texts got results
        assert len(results) == 25
        
        # Check that the API was called exactly 3 times (for 3 batches)
        assert mock_client.chat.completions.create.call_count == 3
        
        # Verify the batch contents were correct
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        third_call = mock_client.chat.completions.create.call_args_list[2]
        
        # Check text counts in each batch
        first_batch_text = first_call[1]["messages"][1]["content"]
        second_batch_text = second_call[1]["messages"][1]["content"]
        third_batch_text = third_call[1]["messages"][1]["content"]
        
        assert first_batch_text.count("TEXT ") == 10
        assert second_batch_text.count("TEXT ") == 10
        assert third_batch_text.count("TEXT ") == 5
        
        # Check confidence scores were tracked correctly
        assert len(labeler.confidence_scores) == 25
        
        # Check first 10 have 0.9 confidence
        for i in range(10):
            assert labeler.confidence_scores[i] == 0.9
            
        # Check second 10 have 0.7 confidence
        for i in range(10, 20):
            assert labeler.confidence_scores[i] == 0.7
            
        # Check last 5 have 0.5 confidence
        for i in range(20, 25):
            assert labeler.confidence_scores[i] == 0.5

    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_integrated_caching_and_deduplication(self, mock_openai, similar_texts):
        """Test integration of caching and deduplication working together."""
        # Setup mock OpenAI
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        # Mock a response for the first batch
        mock_response = """TEXT 1: Python (confidence: HIGH)
TEXT 2: Machine Learning (confidence: HIGH)
TEXT 3: Neural Networks (confidence: HIGH)
TEXT 4: Web Development (confidence: HIGH)"""
        
        mock_response_obj = MagicMock()
        mock_response_obj.choices = [MagicMock()]
        mock_response_obj.choices[0].message.content = mock_response
        mock_client.chat.completions.create.return_value = mock_response_obj
        
        # Create temporary directory for caching
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create labeler with both caching and deduplication
            labeler = LLMTopicLabeler(
                model_type="openai",
                deduplicate=True,
                deduplication_threshold=0.7,
                enable_cache=True,
                cache_dir=tmpdir
            )
            
            # First call - should deduplicate and cache
            first_results = labeler.classify_texts(similar_texts, progress_bar=False)
            
            # Verify called once with deduplicated texts
            assert mock_client.chat.completions.create.call_count == 1
            
            # Reset mock
            mock_client.chat.completions.create.reset_mock()
            
            # Second call with the same texts - should use cache entirely
            second_results = labeler.classify_texts(similar_texts, progress_bar=False)
            
            # Verify no API calls were made
            assert mock_client.chat.completions.create.call_count == 0
            
            # Results should be identical
            assert first_results == second_results
            
            # Now create a modified list with some new texts
            modified_texts = similar_texts.copy()
            modified_texts.append("A completely new text about biology")
            modified_texts.append("Another new text about chemistry")
            
            # Mock response for the new texts only
            new_response = """TEXT 1: Biology (confidence: HIGH)
TEXT 2: Chemistry (confidence: HIGH)"""
            
            new_response_obj = MagicMock()
            new_response_obj.choices = [MagicMock()]
            new_response_obj.choices[0].message.content = new_response
            mock_client.chat.completions.create.return_value = new_response_obj
            
            # Third call with modified list
            third_results = labeler.classify_texts(modified_texts, progress_bar=False)
            
            # Verify API was called once for the new texts only
            assert mock_client.chat.completions.create.call_count == 1
            call_args = mock_client.chat.completions.create.call_args[1]
            
            # Should only have 2 texts in the prompt (the new ones)
            assert call_args["messages"][1]["content"].count("TEXT ") == 2
            
            # Check results length matches the input
            assert len(third_results) == len(modified_texts)
            
            # The first 7 results should match the original results
            for i in range(len(similar_texts)):
                assert third_results[i] == first_results[i]
                
            # The new results should be Biology and Chemistry
            assert third_results[-2] == "Biology"
            assert third_results[-1] == "Chemistry"

    def test_actual_deduplication_algorithm(self):
        """Test that the actual deduplication algorithm works as expected."""
        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
            with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
                with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                    with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                        # Mock dependencies
                        mock_tokenizer.from_pretrained.return_value = MagicMock()
                        mock_model.from_pretrained.return_value = MagicMock()
                        mock_pipeline.return_value = MagicMock()
                        
                        # Create a labeler for testing the helper methods
                        labeler = LLMTopicLabeler(
                            model_type="local",
                            deduplicate=True,
                            deduplication_threshold=0.7  # Lower threshold to detect more duplicates
                        )
                        
                        # Test text similarity calculation
                        sim1 = labeler._calculate_text_similarity(
                            "Python is a programming language", 
                            "Python programming language is versatile"
                        )
                        assert sim1 > 0.7  # Should be similar
                        
                        sim2 = labeler._calculate_text_similarity(
                            "Python is a programming language", 
                            "JavaScript is used for web development"
                        )
                        assert sim2 < 0.5  # Should not be similar
                        
                        # Test the actual deduplication function
                        texts = [
                            "Python is a programming language",
                            "Python programming language is popular",  # Similar to first
                            "JavaScript is a web language",
                            "Machine learning is a subset of AI",
                            "AI and machine learning are related fields"  # Similar to fourth
                        ]
                        
                        duplicate_map = labeler._identify_fuzzy_duplicates(texts)
                        
                        # Should find at least two duplicates
                        assert len(duplicate_map) >= 2
                        
                        # Check specific mappings (the exact indices depend on implementation)
                        # But we can verify the similarity of the mapped texts
                        for dup_idx, original_idx in duplicate_map.items():
                            # The mapped texts should be similar
                            similarity = labeler._calculate_text_similarity(
                                texts[dup_idx], texts[original_idx]
                            )
                            assert similarity >= 0.7


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])