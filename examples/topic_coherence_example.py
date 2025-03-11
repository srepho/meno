#!/usr/bin/env python
"""
Example of using the new topic coherence metrics functionality.
This script demonstrates how to calculate and display topic coherence metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import Meno components
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.coherence import calculate_generic_coherence
from meno.reporting.html_generator import generate_html_report

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Create sample data
def create_sample_data(n_docs=200):
    """Create sample data with clearly defined topics."""
    np.random.seed(42)
    
    # Define topic keywords
    topics = {
        "Technology": ["computer", "software", "hardware", "internet", "digital", 
                      "technology", "code", "programming", "device", "application"],
        "Finance": ["market", "stock", "investment", "financial", "bank", 
                   "money", "fund", "trading", "economy", "asset"],
        "Healthcare": ["medical", "health", "patient", "doctor", "treatment", 
                      "hospital", "disease", "clinical", "therapy", "physician"],
        "Education": ["school", "student", "learning", "teaching", "education", 
                     "academic", "university", "college", "classroom", "course"]
    }
    
    # Generate documents for each topic
    documents = []
    topic_assignments = []
    
    for topic_name, keywords in topics.items():
        # Generate documents for this topic
        for i in range(n_docs // len(topics)):
            # Choose 5-8 keywords from the topic
            n_keywords = np.random.randint(5, 9)
            selected_keywords = np.random.choice(keywords, size=n_keywords, replace=False)
            
            # Add some random words (noise)
            noise_words = ["the", "and", "is", "of", "to", "in", "that", "for", "it", "with", "as", "on"]
            n_noise = np.random.randint(10, 20)
            selected_noise = np.random.choice(noise_words, size=n_noise, replace=True)
            
            # Create document text with mostly the topic keywords
            all_words = list(selected_keywords) * 3 + list(selected_noise)
            np.random.shuffle(all_words)
            doc_text = " ".join(all_words)
            
            documents.append(doc_text)
            topic_assignments.append(topic_name)
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": documents,
        "topic": topic_assignments
    })
    
    return df

# Create output directory
output_dir = Path("examples/output/coherence_example")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate sample data
print("Generating sample data...")
data_df = create_sample_data(n_docs=200)

# Preprocess text
print("Preprocessing text...")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    return tokens

# Tokenize texts
tokenized_texts = [preprocess_text(text) for text in data_df["text"]]

# Train BERTopic model
print("Training BERTopic model...")
topic_model = BERTopicModel(min_topic_size=5)
topic_model.fit(data_df["text"])

# Get topic info
topic_info = topic_model.get_topic_info()
print(f"Discovered {len(topic_info) - 1} topics (excluding outliers)")

# Calculate coherence metrics
print("Calculating coherence metrics...")
coherence_results = topic_model.calculate_coherence(
    texts=tokenized_texts,
    coherence="all",
    top_n=10
)

# Print coherence metrics
print("\nTopic Coherence Metrics:")
for metric_name, score in coherence_results.items():
    print(f"- {metric_name}: {score:.4f}")

# Add coherence to topic assignments
topic_assignments = pd.DataFrame({"doc_id": range(len(data_df))})
topic_assignments["topic"] = topic_model.transform(data_df["text"])[0]
topic_assignments["topic_probability"] = 0.9  # Placeholder

# Add coherence metrics to the topic_assignments DataFrame
for metric_name, score in coherence_results.items():
    topic_assignments[f"coherence_{metric_name}"] = score

# Add general coherence column (use c_v as the main metric)
topic_assignments["coherence"] = coherence_results.get("c_v", 0.0)

# Generate HTML report with coherence metrics
print("\nGenerating HTML report...")
report_path = generate_html_report(
    documents=data_df,
    topic_assignments=topic_assignments,
    output_path=output_dir / "coherence_report.html",
    config={
        "title": "Topic Modeling with Coherence Metrics",
        "include_interactive": True,
        "max_examples_per_topic": 3,
        "include_raw_data": True,
    }
)

print(f"Report generated at: {report_path}")

# Create a bar chart of coherence metrics
plt.figure(figsize=(10, 6))
metrics = []
scores = []

for metric, score in coherence_results.items():
    if score is not None:  # Skip None values
        metrics.append(metric)
        scores.append(score)

bars = plt.bar(metrics, scores, color='skyblue')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.title('Topic Coherence Metrics Comparison')
plt.xlabel('Metric Type')
plt.ylabel('Coherence Score')
plt.xticks(rotation=45)

# Add score labels on bars
for bar in bars:
    height = bar.get_height()
    offset = 0.1 if height >= 0 else -0.1
    va = 'bottom' if height >= 0 else 'top'
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + offset,
        f'{height:.3f}',
        ha='center',
        va=va,
        fontsize=9
    )

plt.tight_layout()
plt.savefig(output_dir / 'coherence_metrics_comparison.png')
plt.close()

print(f"Chart saved to: {output_dir / 'coherence_metrics_comparison.png'}")
print("\nDone!")