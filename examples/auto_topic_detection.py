"""
Example showing automatic topic number detection in Meno.

This example demonstrates how to use Meno's automatic topic detection capability,
allowing the models to determine the optimal number of topics based on the data.
"""

import pandas as pd
import numpy as np
from meno.modeling.unified_topic_modeling import create_topic_modeler
from meno.modeling.embeddings import DocumentEmbedding
from meno.visualization.static_plots import plot_topic_distribution  # Replace with an existing function
from meno.reporting.html_generator import generate_html_report
from pathlib import Path


# Sample data
documents = [
    "Machine learning algorithms require significant computational resources",
    "Deep neural networks have revolutionized computer vision tasks",
    "Natural language processing enables machines to understand text",
    "Reinforcement learning is used to train game-playing AI",
    "Data science relies on statistics and domain knowledge",
    "Feature engineering improves model performance significantly",
    "Transformers have improved natural language understanding",
    "Computer vision systems can now recognize objects in images",
    "Statistical models help understand patterns in data",
    "Unsupervised learning finds patterns without labeled data",
    "Cloud computing provides scalable resources for AI workloads",
    "GPUs accelerate deep learning model training times",
    "Transfer learning applies knowledge from one domain to another",
    "Convolutional neural networks excel at image recognition tasks",
    "Recurrent neural networks process sequential data effectively",
    "Explainable AI helps understand model decision making",
    "Robotics combines sensing, planning, and control algorithms",
    "Autonomous vehicles use machine learning for navigation",
    "Human-computer interaction studies how people use technology",
    "Quantum computing may accelerate certain machine learning algorithms",
    "Edge computing processes data near its source rather than in the cloud",
    "Federated learning trains models while keeping data on local devices",
    "Blockchain provides a distributed and immutable ledger",
    "Internet of things connects physical devices to the internet",
    "Cybersecurity protects systems from unauthorized access",
    "Database systems organize and store data efficiently",
    "Distributed systems coordinate multiple computers to solve problems",
    "Grid computing harnesses diverse computing resources",
    "Software engineering applies engineering principles to software development",
    "Web development creates websites and web applications",
]

# Convert to pandas Series
documents_series = pd.Series(documents)

print("Demonstrating automatic topic detection in Meno")
print(f"Number of documents: {len(documents_series)}")

# Create embedding model
embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")

# 1. Using BERTopic with automatic topic detection
print("\n1. BERTopic with automatic topic detection")

# Create topic modeler with auto_detect_topics=True 
# Add offline_mode=True to bypass potential import issues
bertopic_model = create_topic_modeler(
    method="bertopic",
    auto_detect_topics=True,  # Enable automatic topic detection
    embedding_model=embedding_model,
    offline_mode=True,        # Enable offline mode to bypass import checks
    config_overrides={
        'min_topic_size': 2,  # Lower threshold to allow more topics
        'n_neighbors': 5,     # Adjust UMAP parameters for small dataset
        'verbose': True
    }
)

# Fit the model
bertopic_model.fit(documents_series)

# Check how many topics were detected
topic_info = bertopic_model.get_topic_info()
print(f"BERTopic automatically detected {len(topic_info) - 1} topics")  # -1 for outlier topic
print(f"Topics: {list(topic_info['Topic'])}")

# Display topic keywords
for topic_id in topic_info['Topic']:
    if topic_id != -1:  # Skip outlier topic
        words = topic_info[topic_info['Topic'] == topic_id]['Representation'].values[0]
        print(f"Topic {topic_id}: {words}")
        
# 2. Using Top2Vec with automatic topic detection
print("\n2. Top2Vec with automatic topic detection")

# Create Top2Vec model with auto_detect_topics=True
top2vec_model = create_topic_modeler(
    method="top2vec",
    auto_detect_topics=True,  # Enable automatic topic detection
    embedding_model=embedding_model,
    config_overrides={
        'hdbscan_args': {
            'min_cluster_size': 2,  # Lower threshold for small dataset
            'min_samples': 1
        },
        'umap_args': {
            'n_neighbors': 5,
            'n_components': 2
        }
    }
)

# Fit the model
top2vec_model.fit(documents_series)

# Check how many topics were detected
topic_info = top2vec_model.get_topic_info()
print(f"Top2Vec automatically detected {len(topic_info)} topics")
print(f"Topics: {list(topic_info['Topic'])}")

# Display topic keywords
for i, row in topic_info.iterrows():
    print(f"Topic {row['Topic']}: {row['Representation']}")

# 3. Using the unified interface
print("\n3. Using UnifiedTopicModeler with automatic topic detection")

# Create unified topic modeler
unified_model = create_topic_modeler(
    method="embedding_cluster",  # Uses BERTopic by default
    auto_detect_topics=True,
    embedding_model=embedding_model,
    config_overrides={
        'min_topic_size': 2,
        'n_neighbors': 5
    }
)

# Fit the model
unified_model.fit(documents_series)

# Check how many topics were detected
topic_info = unified_model.get_topic_info()
print(f"UnifiedTopicModeler automatically detected {len(topic_info) - 1} topics")  # -1 for outlier topic
print(f"Topics: {list(topic_info['Topic'])}")

# Create visualization
try:
    # Use a simpler visualization since create_topic_visualization is not available
    topics = unified_model.transform(documents_series)[0]
    fig = plot_topic_distribution(topics)
    print("\nTopic distribution visualization created successfully. You can display it in a notebook or save it.")
except Exception as e:
    print(f"Could not create visualization due to: {e}")

# 4. Demonstrating the open_browser=False default
print("\n4. Generating HTML report with open_browser=False (default)")

# Create output directory
output_dir = Path("examples/output/auto_topic")
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare document DataFrame
docs_df = pd.DataFrame({
    'text': documents_series,
    'topic': unified_model.transform(documents_series)[0]
})

# Generate topic assignments DataFrame
topics, probs = unified_model.transform(documents_series)
topic_assignments = pd.DataFrame({
    'doc_id': range(len(docs_df)),
    'topic': topics,
    'topic_probability': np.max(probs, axis=1) if len(probs.shape) > 1 else probs
})

# Generate HTML report without opening the browser
report_path = generate_html_report(
    documents=docs_df,
    topic_assignments=topic_assignments,
    output_path=output_dir / "auto_topics_report.html",
    config={
        "title": "Auto-Detected Topics Report",
        "include_interactive": True,
        "include_raw_data": True
    },
    # open_browser parameter defaults to False so browser won't open automatically
)

print(f"Report generated at: {report_path}")
print("Notice the browser does not open automatically (new default behavior)")
print("You can open the report manually or set open_browser=True to open automatically")

print("\nAuto-detection is useful when:")
print("1. You have no prior knowledge about how many topics to expect")
print("2. You want the model to find natural clusters in your data")
print("3. You're doing exploratory analysis of an unknown corpus")
print("4. You want to compare with a predetermined number of topics")

# Show how documents are distributed across topics:

print("\nDocument distribution across topics:")
topic_assignments = unified_model.transform(documents_series)
topic_counts = pd.Series(topic_assignments).value_counts()
for topic, count in topic_counts.items():
    percentage = (count / len(documents_series)) * 100
    print(f"Topic {topic}: {count} documents ({percentage:.1f}%)")