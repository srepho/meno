"""
Example of using the enhanced LLM Topic Labeler with batch processing, caching,
confidence scores, and deduplication to optimize API usage.

This example demonstrates how to efficiently process large datasets
of texts for classification using OpenAI's API with optimizations.
"""

import pandas as pd
import time
import os
import argparse
from pathlib import Path

from meno.modeling.llm_topic_labeling import LLMTopicLabeler


def main(api_key=None, azure_endpoint=None, custom_categories=False, deduplicate=True):
    """Run an example of batch classification with LLM Topic Labeler."""
    print("\n==== Meno LLM Topic Labeler Batch Processing Example ====\n")
    
    # Create a sample dataset (or load your own)
    sample_data = pd.DataFrame({
        "id": range(100),
        "text": [
            "New AI model can generate realistic images from text descriptions",
            "The stock market rose 2% after the Fed announcement on interest rates",
            "New study shows link between diet and heart disease prevention",
            "Software developers released a new version with bug fixes and performance improvements",
            "The pharmaceutical company announced promising results in clinical trials",
            "Global tech companies face increased regulatory scrutiny over data privacy",
            "The movie premiere attracted celebrities and fans alike",
            "Scientists discover new species in deep ocean exploration",
            "Investment firm recommends diversifying portfolios amid market volatility",
            "Smartphone manufacturer unveils new device with enhanced camera features",
        ] * 10  # Duplicate each 10 times to simulate larger dataset with repetition
    })
    
    # Set up API configuration
    model_kwargs = {
        "model_type": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_new_tokens": 50,
        "temperature": 0.2,
        "enable_fallback": True,
        "requests_per_minute": 60,  # Respect OpenAI rate limits
        "max_parallel_requests": 4,
        "batch_size": 20,           # Process 20 texts per API call
        "deduplicate": deduplicate,  # Enable deduplication for similar texts
        "deduplication_threshold": 0.9,
        "enable_cache": True,       # Cache results to disk
        "cache_dir": "./.meno_cache",
        "verbose": True,
    }
    
    # Add API key if provided
    if api_key:
        model_kwargs["openai_api_key"] = api_key
        
    # Add Azure endpoint if provided
    if azure_endpoint:
        model_kwargs["api_endpoint"] = azure_endpoint
        model_kwargs["api_version"] = "2023-07-01-preview"
        
    # Create the labeler
    if custom_categories:
        # Use predefined categories
        categories = ["Technology", "Business", "Health", "Entertainment", "Science"]
        system_prompt = f"You are a text classifier that categorizes text into exactly one of these categories: {', '.join(categories)}."
        user_prompt = "Classify the following text into the most appropriate category: {{text}}"
        
        model_kwargs["system_prompt_template"] = system_prompt
        model_kwargs["user_prompt_template"] = user_prompt
        
        labeler = LLMTopicLabeler(**model_kwargs)
        
        print(f"Using predefined categories: {categories}")
    else:
        # Open classification (let the LLM decide appropriate topics)
        system_prompt = "You are an expert at categorizing content into the most relevant and specific topic."
        user_prompt = "Assign a brief, descriptive topic label (1-3 words) to this text: {{text}}"
        
        model_kwargs["system_prompt_template"] = system_prompt
        model_kwargs["user_prompt_template"] = user_prompt
        
        labeler = LLMTopicLabeler(**model_kwargs)
        
        print("Using open classification (no predefined categories)")
    
    # Measure performance
    start_time = time.time()
    
    # Classify all texts
    print(f"\nClassifying {len(sample_data)} texts...")
    results = labeler.classify_texts(
        sample_data["text"].tolist(),
        categories=categories if custom_categories else None,
        progress_bar=True
    )
    
    # Get confidence scores
    confidence_scores = labeler.confidence_scores
    
    # Add results back to dataframe
    sample_data["topic"] = results
    sample_data["confidence"] = sample_data.index.map(lambda idx: confidence_scores.get(idx, 0.0))
    
    # Calculate performance
    elapsed_time = time.time() - start_time
    texts_per_second = len(sample_data) / elapsed_time
    
    # Print statistics
    print(f"\nResults Summary:")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Texts per second: {texts_per_second:.2f}")
    print(f"Unique topics found: {sample_data['topic'].nunique()}")
    print(f"Average confidence score: {sample_data['confidence'].mean():.2f}")
    
    # Show topic distribution
    topic_counts = sample_data["topic"].value_counts()
    print("\nTopic Distribution:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} ({count/len(sample_data)*100:.1f}%)")
    
    # Show some examples with confidence scores
    print("\nSample Classifications:")
    sample_results = sample_data.drop_duplicates(subset=["text"]).sample(min(5, len(sample_data)))
    for _, row in sample_results.iterrows():
        print(f"  Text: {row['text'][:50]}...")
        print(f"  Topic: {row['topic']} (confidence: {row['confidence']:.2f})")
        print()
    
    # Save results
    output_dir = Path("./meno_output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "topic_classification_results.csv"
    sample_data.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return sample_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Topic Labeler Batch Processing Example")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--azure", help="Azure OpenAI endpoint URL")
    parser.add_argument("--categories", action="store_true", help="Use predefined categories")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    
    args = parser.parse_args()
    
    main(
        api_key=args.api_key,
        azure_endpoint=args.azure,
        custom_categories=args.categories,
        deduplicate=not args.no_dedupe
    )