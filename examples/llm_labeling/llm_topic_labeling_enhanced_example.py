"""
Enhanced LLM Topic Labeling Example

This example demonstrates the enhanced functionality of the LLM Topic Labeling
system in Meno v1.2.4, including:
- Confidence scores for topic classifications
- Automatic caching of results
- Deduplication of similar texts for efficiency
- Dynamic context window management
"""

import pandas as pd
import numpy as np
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.modeling.bertopic_model import BERTopicModel

# Sample data - in a real application, this would be your document collection
sample_texts = [
    "The new iPhone features advanced AI capabilities for photo editing and organization.",
    "Stock markets fell by 2% today following the central bank's interest rate decision.",
    "Scientists discover new species of deep-sea creatures near hydrothermal vents.",
    "Local government announces plans to build affordable housing in city center.",
    "A new study shows the benefits of Mediterranean diet for heart health and longevity.",
    "The tech company unveiled its latest smartphone with AI capabilities.",  # Similar to first text
    "The stock market declined after the Federal Reserve's announcement.",  # Similar to second text
    "Researchers found previously unknown marine life in the deep ocean.",  # Similar to third text
]

# Create additional similar texts to demonstrate deduplication
expanded_texts = sample_texts.copy()
for text in sample_texts[:4]:
    slightly_modified = text.replace("the", "the") + " Further analysis is ongoing."
    expanded_texts.append(slightly_modified)

# 1. Create the LLM Topic Labeler with enhanced settings

# Azure OpenAI example (default)
azure_labeler = LLMTopicLabeler(
    model_name="your-deployment-name",  # Your Azure deployment name
    api_key="your-api-key",             # Your Azure OpenAI API key
    api_endpoint="https://your-resource.openai.azure.com", # Your Azure endpoint
    api_version="2023-05-15",           # Azure API version
    use_azure=True,                     # Use Azure OpenAI (default)
    
    # Enhanced features
    enable_cache=True,                  # Enable caching for performance
    cache_dir="./.meno_cache",          # Set cache directory
    cache_ttl=86400,                    # Cache TTL in seconds (1 day)
    
    deduplicate=True,                   # Enable deduplication of similar texts
    deduplication_threshold=0.85,       # Set similarity threshold (0-1)
    
    # Customization
    system_prompt_template="You are an expert at categorizing content into appropriate topics."
)

# Standard OpenAI example
openai_labeler = LLMTopicLabeler(
    model_name="gpt-4o",                # OpenAI model name
    api_key="your-openai-api-key",      # Your OpenAI API key
    use_azure=False,                    # Use standard OpenAI API
    
    # Rate limiting for API usage efficiency
    requests_per_minute=60,             # Limit to 60 requests per minute
    
    # Parallelism and batching
    max_parallel_requests=4,            # Process up to 4 requests in parallel
    batch_size=10,                      # Process up to 10 texts in a single API call
    
    # Prompt templates can be customized
    user_prompt_template="Classify the following text into a brief descriptive topic (2-4 words): {{text}}"
)

# For this example, we'll use the standard OpenAI client 
# (replace with your actual API key to run this example)
labeler = LLMTopicLabeler(
    model_name="gpt-3.5-turbo",         # Using GPT-3.5 Turbo
    use_azure=False,                    # Use standard OpenAI API
    deduplicate=True,                   # Enable deduplication
    deduplication_threshold=0.85,       # Set similarity threshold (0-1)
    batch_size=10                       # Process up to 10 texts in a single API call
)

# 2. Demonstrate classification with confidence scores
print("Classifying texts with confidence scores...")
# Using normal progress bar
results = labeler.classify_texts(expanded_texts, progress_bar=True)

# Different ways to use the simple progress tracking if tqdm doesn't work:
# 1. Basic simple progress with default interval (5 items)
# results = labeler.classify_texts(expanded_texts, progress_bar="simple")

# 2. Simple progress with custom interval (print every 2 items)
# results = labeler.classify_texts(expanded_texts, progress_bar={"type": "simple", "interval": 2})

# 3. Simple progress with less frequent updates (print every 10 items)
# results = labeler.classify_texts(expanded_texts, progress_bar={"type": "simple", "interval": 10})

# Get confidence scores
confidence_scores = labeler.confidence_scores

# Display results with confidence
results_df = pd.DataFrame({
    "text": expanded_texts,
    "topic": results,
    "confidence": [confidence_scores.get(i, "N/A") for i in range(len(expanded_texts))]
})

print("\nClassification Results:")
print(results_df[["text", "topic", "confidence"]].head())

# 3. Demonstrate caching benefits
print("\nDemonstrating cache benefits...")
print("First classification (without cache):")
# Clear any existing cache with a new instance
new_labeler = LLMTopicLabeler(
    model_type="openai",
    model_name="gpt-3.5-turbo",
    enable_cache=True,
    deduplicate=False,  # Disable deduplication to test cache only
)

# Time the classification without cache
import time
start_time = time.time()
new_labeler.classify_texts(sample_texts[:3])
first_time = time.time() - start_time

print(f"Time taken without cache: {first_time:.2f} seconds")

# Time the classification with cache
print("Second classification (with cache):")
start_time = time.time()
new_labeler.classify_texts(sample_texts[:3])
second_time = time.time() - start_time

print(f"Time taken with cache: {second_time:.2f} seconds")
if second_time < first_time:
    print(f"Performance improvement: {(1 - second_time/first_time) * 100:.1f}%")

# 4. Demonstrate deduplication benefits
print("\nDemonstrating deduplication benefits...")
# Create a new labeler with deduplication on
dedup_labeler = LLMTopicLabeler(
    model_type="openai",
    model_name="gpt-3.5-turbo",
    deduplicate=True,
    deduplication_threshold=0.85,
    enable_cache=False  # Disable cache to test deduplication only
)

# Time classification with deduplication (the original text has duplicates)
start_time = time.time()
dedup_results = dedup_labeler.classify_texts(expanded_texts)
dedup_time = time.time() - start_time

print(f"Classified {len(expanded_texts)} texts (with duplicates)")
print(f"Time taken with deduplication: {dedup_time:.2f} seconds")

# 5. Integrate with a BERTopic model (if BERTopic is installed)
try:
    print("\nIntegrating with BERTopic...")
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Create a small BERTopic model
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer)
    
    # Fit the model (this creates topics)
    topics, probs = topic_model.fit_transform(sample_texts)
    
    # Use LLM to label topics
    print("Generating LLM topic labels...")
    topic_labels = labeler.label_topics(topic_model=topic_model)
    
    print("\nGenerated topic labels:")
    for topic_id, label in topic_labels.items():
        if topic_id != -1:  # Skip outlier topic
            words = [word for word, _ in topic_model.get_topic(topic_id)][:5]
            print(f"Topic {topic_id}: {label} - Keywords: {', '.join(words)}")
            
    # Update the model with these labels
    updated_model = labeler.update_model_topic_names(topic_model=topic_model)
    
except ImportError:
    print("BERTopic not installed, skipping topic model integration example")

print("\nEnhanced LLM Topic Labeling example completed.")