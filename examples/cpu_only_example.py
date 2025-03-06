"""
Example demonstrating meno's CPU-first design for topic modeling.

This example showcases how to use meno for topic modeling on a machine without
GPU acceleration, demonstrating its efficient CPU-only operation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from time import time

# Add parent directory to path to import meno
sys.path.append(str(Path(__file__).parent.parent))
from meno import MenoTopicModeler
from meno.modeling.embeddings import DocumentEmbedding

# Sample data generation
def generate_sample_data(n_samples=500):
    """Generate synthetic data for demonstration."""
    print(f"Generating {n_samples} sample documents...")
    
    # Create topic templates
    topics = {
        "Technology": [
            "artificial intelligence machine learning data algorithms computers",
            "software development programming code application web system",
            "cloud computing storage server infrastructure network security",
            "mobile devices apps smartphones tablets technology hardware"
        ],
        "Healthcare": [
            "medical health doctors patients hospital treatment therapy",
            "disease diagnosis symptoms medication prescription clinical",
            "healthcare insurance coverage benefits claims provider",
            "wellness prevention fitness nutrition exercise lifestyle"
        ],
        "Finance": [
            "investment market stocks bonds trading portfolio assets",
            "banking financial loans credit mortgage debt interest rates",
            "retirement savings pension fund planning wealth management",
            "insurance risk coverage policy premium claims benefits"
        ],
        "Education": [
            "learning teaching students school curriculum education classroom",
            "academic university college degree research scholarship campus",
            "training skills development career professional certification",
            "online courses e-learning digital education virtual classroom"
        ]
    }
    
    # Generate documents from topics
    documents = []
    doc_ids = []
    doc_topics = []
    
    topic_names = list(topics.keys())
    doc_id = 1
    
    for _ in range(n_samples):
        # Select a random topic
        topic = np.random.choice(topic_names)
        doc_topics.append(topic)
        
        # Select a random template
        template = np.random.choice(topics[topic])
        
        # Create variations by adding noise and varying length
        words = template.split()
        
        # Add some noise and vary length
        num_words = len(words) + np.random.randint(-3, 10)
        if num_words < 5:
            num_words = 5
            
        # Select random words with replacement to create variations
        selected_words = np.random.choice(words, size=num_words, replace=True)
        
        # Add some random transitional words
        transitions = ["and", "also", "including", "with", "for", "about", "regarding"]
        for i in range(2, len(selected_words), 5):
            if i < len(selected_words):
                selected_words[i] = np.random.choice(transitions)
        
        document = " ".join(selected_words)
        documents.append(document)
        doc_ids.append(f"doc_{doc_id}")
        doc_id += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": documents,
        "id": doc_ids,
        "actual_topic": doc_topics
    })
    
    print(f"Generated {len(df)} documents across {len(topic_names)} topics")
    return df

# Define configuration with explicit CPU settings
CPU_CONFIG = {
    "preprocessing": {
        "normalization": {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": False,
            "lemmatize": True,
            "language": "en",
        },
        "stopwords": {
            "use_default": True,
            "additional": ["and", "also", "including", "with", "for", "about", "regarding"],
        },
    },
    "modeling": {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",  # Small, fast model for CPU
            "device": "cpu",                   # Explicitly use CPU
            "use_gpu": False,                  # Disable GPU
            "batch_size": 32,                  # Batch size optimized for CPU
        },
        "topic_matching": {
            "threshold": 0.15,
            "assign_multiple": True,
            "max_topics_per_doc": 2,
        },
    },
    "visualization": {
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
        },
        "plots": {
            "width": 900,
            "height": 700,
        },
    },
}

# Define manual topics and descriptions
TOPICS = ["Technology", "Healthcare", "Finance", "Education"]

TOPIC_DESCRIPTIONS = [
    "Technology, computing, software, hardware, AI, and digital innovations",
    "Healthcare, medical, doctors, patients, treatments, and wellness",
    "Finance, banking, investments, money, markets, and economics",
    "Education, learning, teaching, schools, universities, and training"
]

def main():
    """Run the CPU-optimized topic modeling process."""
    # Create output directory
    output_dir = Path("cpu_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    df = generate_sample_data(n_samples=500)
    
    # Save sample data for reference
    df.to_csv(output_dir / "sample_data.csv", index=False)
    
    print("\n--- CPU-Based Topic Modeling Demo ---")
    
    # Demonstrate that we're using CPU-only
    print("\nChecking CPU usage:")
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",  # Small, fast model
        use_gpu=False,                  # Explicitly disable GPU
        device="cpu"                    # Explicitly use CPU
    )
    
    print(f"Using device: {embedding_model.device}")
    print(f"GPU available but not used: {embedding_model.gpu_available}")
    print(f"Model loaded on: {embedding_model.model.device}")
    
    # Initialize topic modeler with CPU config
    print("\nInitializing topic modeler with CPU configuration...")
    start_time = time()
    topic_modeler = MenoTopicModeler(config_overrides=CPU_CONFIG)
    
    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = topic_modeler.preprocess(df, text_column="text", id_column="id")
    
    # Unsupervised topic discovery
    print("\n=== Unsupervised Topic Discovery (CPU) ===")
    print("Embedding documents on CPU...")
    topic_modeler.embed_documents()
    
    print("Discovering topics using clustering...")
    unsupervised_results = topic_modeler.discover_topics(
        method="embedding_cluster", 
        num_topics=len(TOPICS)
    )
    
    print("Top topics discovered:")
    topic_counts = unsupervised_results["topic"].value_counts()
    print(topic_counts.head(10))
    
    print("Creating visualization...")
    fig = topic_modeler.visualize_embeddings(return_figure=True)
    fig.write_html(output_dir / "unsupervised_cpu_visualization.html")
    
    # Supervised topic matching
    print("\n=== Supervised Topic Matching (CPU) ===")
    # Use the same embeddings
    
    print("Matching documents to predefined topics...")
    supervised_results = topic_modeler.match_topics(
        topics=TOPICS,
        descriptions=TOPIC_DESCRIPTIONS,
        threshold=0.15,
    )
    
    print("Topic distribution:")
    topic_distribution = supervised_results["topic"].value_counts()
    for topic, count in topic_distribution.items():
        print(f"- {topic}: {count} documents ({count/len(supervised_results)*100:.1f}%)")
    
    print("Creating visualizations...")
    fig = topic_modeler.visualize_topic_distribution(return_figure=True)
    fig.write_html(output_dir / "cpu_topic_distribution.html")
    
    fig = topic_modeler.visualize_embeddings(return_figure=True)
    fig.write_html(output_dir / "supervised_cpu_visualization.html")
    
    # Generate report
    print("\nGenerating HTML report...")
    report_path = topic_modeler.generate_report(
        output_path=output_dir / "cpu_topics_report.html",
        include_interactive=True,
        title="CPU-Based Topic Modeling Report"
    )
    print(f"Report generated at {report_path}")
    
    # Calculate performance metrics
    end_time = time()
    total_time = end_time - start_time
    print(f"\nPerformance metrics (CPU):")
    print(f"- Total processing time: {total_time:.2f} seconds")
    print(f"- Documents per second: {len(df)/total_time:.2f}")
    
    # Compare with ground truth (actual_topic)
    if "actual_topic" in df.columns:
        print("\nEvaluating against ground truth:")
        # Check if each document was assigned to its actual topic
        merged = supervised_results.merge(df[["id", "actual_topic"]], on="id")
        correct = 0
        for _, row in merged.iterrows():
            if row["topic"] == row["actual_topic"]:
                correct += 1
        accuracy = correct / len(merged) * 100
        print(f"- Accuracy: {accuracy:.2f}%")
    
    print("\nCPU-based topic modeling completed successfully!")

if __name__ == "__main__":
    main()