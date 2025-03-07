"""Simple demonstration of the new lightweight components.

This script shows the basic usage of the lightweight models and visualizations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

# Import lightweight models
from meno.modeling.simple_models.lightweight_models import (
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel
)

# Sample documents (simplified to avoid potential issues)
sample_documents = [
    "Machine learning uses statistics to enable computers to learn from data",
    "Deep learning uses neural networks with many layers",
    "Neural networks are computing systems inspired by biological brains",
    "Python is a popular language for data science and machine learning",
    "TensorFlow and PyTorch are popular deep learning frameworks",
    "Natural language processing enables understanding human language",
    "Computer vision helps computers interpret images and videos",
    "Healthcare technology uses AI to improve diagnostics",
    "Medical imaging uses computer vision for medical scans",
    "Electronic health records store patient data digitally",
    "Climate change refers to shifts in global temperature patterns",
    "Renewable energy sources help reduce carbon emissions",
    "Sustainable development aims to preserve the environment",
    "Conservation efforts focus on protecting biodiversity",
    "Electric vehicles reduce reliance on fossil fuels"
]

print(f"Number of documents: {len(sample_documents)}")
print("First document: " + sample_documents[0][:50] + "...")
print("-" * 40)

# Create and fit TF-IDF topic model (most lightweight option)
print("\nCreating TF-IDF topic model...")
tfidf_model = TFIDFTopicModel(num_topics=3, random_state=42)
tfidf_model.fit(sample_documents)

# Get topic information
topic_info = tfidf_model.get_topic_info()
print("\nDiscovered topics:")
for _, row in topic_info.iterrows():
    print(f"  Topic {row['Topic']}: {row['Name']}")
    
# Get document assignments
doc_info = tfidf_model.get_document_info()
print("\nSample document assignments:")
for i in range(min(5, len(doc_info))):
    print(f"  Document {i}: Topic {doc_info.iloc[i]['Topic']} ({doc_info.iloc[i]['Name']})")

# NMF model (another lightweight alternative)
print("\nCreating NMF topic model...")
nmf_model = NMFTopicModel(num_topics=3, random_state=42)
nmf_model.fit(sample_documents)

# LSA model
print("\nCreating LSA topic model...")
lsa_model = LSATopicModel(num_topics=3, random_state=42)
lsa_model.fit(sample_documents)

# Compare topic distributions
print("\nComparing topic distributions:")
for model_name, model in [("TF-IDF", tfidf_model), ("NMF", nmf_model), ("LSA", lsa_model)]:
    topic_info = model.get_topic_info()
    print(f"\n{model_name} model:")
    for _, row in topic_info.iterrows():
        print(f"  Topic {row['Topic']} ({row['Name']}): {row['Count']} documents")

# Working with topics
for topic_id in range(3):
    words = tfidf_model.get_topic(topic_id)
    if words:
        print(f"\nTop words for Topic {topic_id}:")
        for word, weight in words[:5]:
            print(f"  {word}: {weight:.3f}")

# Test transform
new_docs = ["AI is transforming many industries"]
print("\nClassifying new document:")
print(f"  \"{new_docs[0]}\"")

for model_name, model in [("TF-IDF", tfidf_model), ("NMF", nmf_model), ("LSA", lsa_model)]:
    result = model.transform(new_docs)
    
    if model_name == "TF-IDF":
        assignments, probs = result
        topic_id = assignments[0]
        print(f"  {model_name} model: Topic {topic_id} ({model.topics[topic_id]})")
    else:
        doc_topic_matrix = result
        topic_id = np.argmax(doc_topic_matrix[0])
        print(f"  {model_name} model: Topic {topic_id} ({model.topics[topic_id]})")

print("\nSimple integrated demo completed successfully!")
print("This demonstrates that the lightweight models work correctly together.")