"""
Advanced LLM Topic Labeling Example

This example demonstrates advanced usage patterns for the LLM Topic Labeling
system in Meno v1.2.4, including:
- Custom prompt templates for domain-specific classification
- Dynamic batch size adjustment based on text length
- Integration with topic feedback mechanisms
- Hybrid classification with predefined categories and LLM suggestions
"""

import pandas as pd
import numpy as np
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.visualization.enhanced_viz.feedback_viz import plot_feedback_impact
import os
import time
import logging

# Set up logging for visibility
logging.basicConfig(level=logging.INFO)

# Load sample data - in a real application, this would be your dataset
# For this example, we'll create a synthetic dataset
def generate_sample_data(size=100):
    # Generate synthetic topics
    topics = [
        "Technology and AI",
        "Financial Markets",
        "Healthcare and Medicine",
        "Politics and Governance",
        "Environment and Climate",
        "Entertainment and Media",
        "Education and Learning",
        "Sports and Athletics"
    ]
    
    # Generate sample texts for each topic
    data = []
    for i in range(size):
        topic_idx = i % len(topics)
        topic = topics[topic_idx]
        
        # Create sample text based on topic
        if topic == "Technology and AI":
            text = f"New developments in artificial intelligence are transforming how we {np.random.choice(['work', 'live', 'interact', 'communicate'])}. " \
                  f"Companies like {np.random.choice(['Google', 'Microsoft', 'Amazon', 'Apple'])} are investing heavily in {np.random.choice(['machine learning', 'neural networks', 'deep learning', 'natural language processing'])}."
        elif topic == "Financial Markets":
            text = f"The stock market {np.random.choice(['rose', 'fell', 'fluctuated', 'stabilized'])} today after {np.random.choice(['Federal Reserve announcements', 'economic data releases', 'corporate earnings reports', 'global market movements'])}. " \
                  f"Investors are {np.random.choice(['optimistic about', 'concerned about', 'closely watching', 'responding to'])} recent economic trends."
        elif topic == "Healthcare and Medicine":
            text = f"Researchers have {np.random.choice(['discovered', 'developed', 'identified', 'published findings on'])} a new {np.random.choice(['treatment', 'approach', 'understanding', 'breakthrough'])} for {np.random.choice(['cancer', 'heart disease', 'diabetes', 'infectious diseases'])}. " \
                  f"This could lead to {np.random.choice(['better patient outcomes', 'more effective therapies', 'improved quality of life', 'reduced healthcare costs'])}."
        else:
            # Generic text for other topics
            text = f"This is sample text number {i+1} related to {topic}. It contains information that would typically be found in documents about this subject."
        
        data.append({"id": i, "text": text, "true_topic": topic})
    
    return pd.DataFrame(data)

# Generate sample data
print("Generating sample dataset...")
df = generate_sample_data(size=50)
print(f"Generated {len(df)} sample documents")

# 1. Create LLM Labeler with advanced settings
labeler = LLMTopicLabeler(
    model_type="openai",
    model_name="gpt-3.5-turbo",
    
    # Advanced settings
    temperature=0.3,           # Lower temperature for more consistent results
    max_new_tokens=150,        # Allow longer responses
    enable_cache=True,         # Enable caching
    deduplicate=True,          # Enable deduplication
    deduplication_threshold=0.9,  # Higher threshold for stricter deduplication
    requests_per_minute=40,    # Conservative rate limiting
    verbose=True,              # Enable verbose logging
)

# 2. Domain-specific classification with custom prompts
print("\nPerforming domain-specific classification...")

# Define domain-specific prompts
healthcare_system_prompt = """
You are a medical domain expert specializing in healthcare topic classification.
Classify texts into precise medical and healthcare categories, using standard
terminology from the field. Be specific about medical conditions, treatments,
and healthcare policy areas.
"""

healthcare_user_prompt = """
Please classify the following healthcare-related text into a specific medical 
or healthcare topic category (2-5 words):

{{text}}

Classification:
"""

# Filter for healthcare-related texts
healthcare_df = df[df['true_topic'] == 'Healthcare and Medicine']
healthcare_texts = healthcare_df['text'].tolist()

# Perform healthcare-specific classification
healthcare_results = labeler.classify_texts(
    texts=healthcare_texts,
    system_prompt=healthcare_system_prompt,
    user_prompt_template=healthcare_user_prompt
)

print("\nHealthcare Classification Results:")
for i, (text, result) in enumerate(zip(healthcare_texts, healthcare_results)):
    print(f"{i+1}. {text[:100]}... → {result}")

# 3. Classification with predefined categories
print("\nPerforming classification with predefined categories...")

# Define main categories
main_categories = [
    "Technology",
    "Finance",
    "Healthcare",
    "Politics",
    "Environment",
    "Entertainment",
    "Education",
    "Sports",
    "Other"
]

# Sample texts from different domains
mixed_texts = df.sample(10)['text'].tolist()

# Classify with predefined categories
categorized_results = labeler.classify_texts(
    texts=mixed_texts,
    categories=main_categories,
    system_prompt="You are a topic classification system. Classify each text into exactly one of the provided categories.",
)

print("\nPredefined Category Classification Results:")
for i, (text, result) in enumerate(zip(mixed_texts, categorized_results)):
    print(f"{i+1}. {text[:100]}... → {result}")
    
# 4. Adaptive processing based on text length
print("\nDemonstrating adaptive processing based on text length...")

# Create texts of varying lengths
short_texts = [t[:100] for t in df['text'].tolist()[:5]]
medium_texts = df['text'].tolist()[5:10]
long_texts = [t + " " + t for t in df['text'].tolist()[10:15]]  # Duplicate to make longer

# Function to process texts with adaptive batch size
def process_with_adaptive_batch_size(texts):
    # Calculate average token count (rough estimate: 1 token ≈ 4 chars)
    avg_length = sum(len(t) for t in texts) / len(texts)
    avg_tokens = avg_length / 4
    
    # Adjust batch size based on text length
    if avg_tokens < 50:  # Very short texts
        batch_size = 20
    elif avg_tokens < 100:  # Short texts
        batch_size = 15
    elif avg_tokens < 200:  # Medium texts
        batch_size = 10
    else:  # Long texts
        batch_size = 5
    
    print(f"Average text length: {avg_length:.1f} chars (est. {avg_tokens:.1f} tokens)")
    print(f"Using adaptive batch size: {batch_size}")
    
    # Process with adjusted batch size
    start_time = time.time()
    results = labeler.classify_texts(
        texts=texts,
        batch_size=batch_size
    )
    elapsed = time.time() - start_time
    
    return results, elapsed, batch_size

# Process different text lengths
print("\nProcessing short texts:")
short_results, short_time, short_batch = process_with_adaptive_batch_size(short_texts)

print("\nProcessing medium texts:")
medium_results, medium_time, medium_batch = process_with_adaptive_batch_size(medium_texts)

print("\nProcessing long texts:")
long_results, long_time, long_batch = process_with_adaptive_batch_size(long_texts)

# Display efficiency metrics
print("\nEfficiency Metrics:")
print(f"Short texts: {len(short_texts)} processed in {short_time:.2f}s ({short_time/len(short_texts):.2f}s per text, batch size {short_batch})")
print(f"Medium texts: {len(medium_texts)} processed in {medium_time:.2f}s ({medium_time/len(medium_texts):.2f}s per text, batch size {medium_batch})")
print(f"Long texts: {len(long_texts)} processed in {long_time:.2f}s ({long_time/len(long_texts):.2f}s per text, batch size {long_batch})")

# 5. Simulated feedback integration
print("\nDemonstrating feedback integration...")

# Classify initial set of texts
initial_texts = df.sample(20)['text'].tolist()
initial_results = labeler.classify_texts(initial_texts)

# Simulate user feedback (in a real system, this would come from users)
feedback = {
    3: "AI Technology Trends",   # Override classification for document 3
    7: "Financial Markets",      # Override classification for document 7
    12: "Climate Change Policy"  # Override classification for document 12
}

# Create a feedback-aware classification function
def classify_with_feedback(texts, prior_feedback, threshold=0.85):
    """Classify texts with awareness of prior feedback"""
    # First try to use feedback from similar texts
    results = []
    used_feedback = []
    
    for i, text in enumerate(texts):
        feedback_applied = False
        
        # Check if we have feedback for a similar text
        for fb_idx, fb_label in prior_feedback.items():
            if fb_idx < len(initial_texts):  # Ensure feedback index is valid
                feedback_text = initial_texts[fb_idx]
                
                # Calculate similarity (simplified for example)
                similarity = len(set(text.split()).intersection(set(feedback_text.split()))) / \
                             len(set(text.split()).union(set(feedback_text.split())))
                
                if similarity > threshold:
                    results.append(fb_label)
                    used_feedback.append((i, fb_idx, similarity))
                    feedback_applied = True
                    break
        
        if not feedback_applied:
            # If no feedback applies, mark for regular classification
            results.append(None)
    
    # Get indices of texts that need regular classification
    to_classify_indices = [i for i, r in enumerate(results) if r is None]
    texts_to_classify = [texts[i] for i in to_classify_indices]
    
    if texts_to_classify:
        # Classify texts that don't have applicable feedback
        model_results = labeler.classify_texts(texts_to_classify)
        
        # Merge model results with feedback-based results
        for i, model_result in zip(to_classify_indices, model_results):
            results[i] = model_result
    
    return results, used_feedback

# Classify new texts with feedback
new_texts = df.sample(10)['text'].tolist()
feedback_results, used_feedback = classify_with_feedback(new_texts, feedback)

# Display results
print("\nFeedback-aware Classification Results:")
print(f"Applied feedback in {len(used_feedback)} out of {len(new_texts)} classifications")

for i, (text, result) in enumerate(zip(new_texts, feedback_results)):
    # Check if feedback was used
    feedback_info = ""
    for fb_i, fb_idx, fb_sim in used_feedback:
        if fb_i == i:
            feedback_info = f" (used feedback from text {fb_idx}, similarity: {fb_sim:.2f})"
            break
            
    print(f"{i+1}. {text[:80]}... → {result}{feedback_info}")

print("\nAdvanced LLM Topic Labeling example completed.")