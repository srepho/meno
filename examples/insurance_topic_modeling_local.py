"""
Example of using Meno for topic modeling on Australian insurance complaints dataset.
This version uses a locally downloaded dataset instead of loading from Hugging Face.
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add parent directory to path to import meno
sys.path.append(str(Path(__file__).parent.parent))
from meno import MenoTopicModeler

# Load the dataset from local files
def load_insurance_dataset():
    """Load and prepare the Australian insurance dataset from local files."""
    local_path = Path(__file__).parent.parent / "data" / "australian_insurance" / "train.csv"
    
    if not local_path.exists():
        print(f"Error: Local dataset not found at {local_path}")
        print("Please run tools/download_insurance_dataset.py first to download the dataset.")
        sys.exit(1)
    
    print(f"Loading dataset from {local_path}...")
    df = pd.read_csv(local_path)
    
    # Extract annotation types for each document
    annotations_list = []
    for annotations_str in df["annotations"]:
        try:
            annotations_json = json.loads(annotations_str)
            annotation_types = set([a["type"] for a in annotations_json["annotations"]])
            annotations_list.append(list(annotation_types))
        except:
            annotations_list.append([])
    
    df["annotation_types"] = annotations_list
    print(f"Loaded {len(df)} documents")
    return df

# Define manual topics
INSURANCE_TOPICS = [
    "Premium Increases",
    "Claim Delays",
    "Claim Denials",
    "Poor Customer Service",
    "Documentation Issues",
    "Communication Problems",
    "Assessor Complaints",
    "Repair Quality Issues",
    "Payout Amount Disputes",
]

TOPIC_DESCRIPTIONS = [
    "Complaints about insurance premium increases, pricing, or rate hikes",
    "Issues with delays in processing claims or slow response times",
    "Complaints about claims being rejected, denied, or not fully covered",
    "Negative experiences with customer service, staff, or support",
    "Problems with excessive paperwork, documentation requirements, or forms",
    "Issues with lack of communication, updates, or responsiveness",
    "Complaints about assessors, inspections, or claim evaluation process",
    "Problems with repair quality, contractors, or workmanship",
    "Disputes about settlement amounts, compensation, or payout values",
]

# Define configuration overrides
CONFIG_OVERRIDES = {
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
            "additional": ["dear", "insurance", "company", "sincerely", "writing", "regards", "customer"],
        },
    },
    "modeling": {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",  # Small, fast model for demonstration
            "batch_size": 32,
        },
        "topic_matching": {
            "threshold": 0.15,  # Lower threshold for more topic assignments
            "assign_multiple": True,
            "max_topics_per_doc": 3,
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

def main():
    """Run the topic modeling process."""
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    df = load_insurance_dataset()
    
    # Initialize topic modeler
    print("Initializing topic modeler...")
    topic_modeler = MenoTopicModeler(config_overrides=CONFIG_OVERRIDES)
    
    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = topic_modeler.preprocess(df, text_column="text", id_column="id")
    
    # Method 1: Unsupervised topic discovery
    print("\n=== Unsupervised Topic Discovery ===")
    print("Embedding documents...")
    topic_modeler.embed_documents()
    
    print("Discovering topics using clustering...")
    unsupervised_results = topic_modeler.discover_topics(
        method="embedding_cluster", 
        num_topics=len(INSURANCE_TOPICS)
    )
    
    print("Top topics discovered:")
    topic_counts = unsupervised_results["topic"].value_counts()
    print(topic_counts.head(10))
    
    print("Creating visualization...")
    fig = topic_modeler.visualize_embeddings(return_figure=True)
    fig.write_html(output_dir / "unsupervised_topics_visualization.html")
    
    # Method 2: Supervised topic matching
    print("\n=== Supervised Topic Matching ===")
    # Use the same embeddings from above
    
    print("Matching documents to predefined topics...")
    supervised_results = topic_modeler.match_topics(
        topics=INSURANCE_TOPICS,
        descriptions=TOPIC_DESCRIPTIONS,
        threshold=0.15,
    )
    
    print("Topic distribution:")
    topic_distribution = supervised_results["topic"].value_counts()
    for topic, count in topic_distribution.items():
        print(f"- {topic}: {count} documents ({count/len(supervised_results)*100:.1f}%)")
    
    print("Creating visualization...")
    fig = topic_modeler.visualize_topic_distribution(return_figure=True)
    fig.write_html(output_dir / "topic_distribution.html")
    
    fig = topic_modeler.visualize_embeddings(return_figure=True)
    fig.write_html(output_dir / "supervised_topics_visualization.html")
    
    # Generate report
    print("\nGenerating HTML report...")
    report_path = topic_modeler.generate_report(
        output_path=output_dir / "insurance_topics_report.html",
        include_interactive=True
    )
    print(f"Report generated at {report_path}")
    
    # Export results
    print("Exporting results...")
    export_paths = topic_modeler.export_results(
        output_path=output_dir / "insurance_results",
        formats=["csv", "json"],
        include_embeddings=False
    )
    print(f"Results exported to {export_paths}")

if __name__ == "__main__":
    main()