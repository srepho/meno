"""
Generate a sample topic modeling report for documentation purposes.

This script creates a sample report using synthetic data to demonstrate
the reporting capabilities of the meno package.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import meno
sys.path.append(str(Path(__file__).parent.parent))
from meno.meno import MenoTopicModeler

# Create output directory
SAMPLE_DIR = Path(__file__).parent / "sample_reports"
SAMPLE_DIR.mkdir(exist_ok=True)

def generate_sample_data(n_samples=100):
    """Generate synthetic data for demonstration."""
    print(f"Generating {n_samples} sample documents...")
    
    # Create topic templates
    topics = {
        "Customer Service": [
            "poor service customer representative issue complaint resolution unhelpful",
            "rude staff service experience terrible customer support unacceptable",
            "wait times customer service long delays frustrating experience",
            "spoke manager complaint service issue resolved finally"
        ],
        "Claims Process": [
            "claim denied insurance coverage policy terms exclusion rejected",
            "claim process slow delayed payment waiting time consuming",
            "documentation claims process excessive paperwork complicated difficult",
            "claim rejected unfairly policy coverage insurance dispute appeal"
        ],
        "Billing Issues": [
            "overcharged bill incorrect amount charge error refund billing",
            "automatic payment failed charged twice billing error bank",
            "bill statement confusing unclear charges fees explanation missing",
            "unexpected fee charge bill statement surprise increase"
        ],
        "Policy Coverage": [
            "coverage inadequate policy insufficient protection limits too low",
            "misled coverage salesperson policy promised expected different",
            "coverage changed without notice policy terms conditions modified",
            "exclusion policy fine print coverage limitation surprised"
        ]
    }
    
    # Generate documents
    documents = []
    doc_topics = []
    
    topic_names = list(topics.keys())
    
    for _ in range(n_samples):
        # Select a random topic
        topic = np.random.choice(topic_names)
        doc_topics.append(topic)
        
        # Select a random template
        template = np.random.choice(topics[topic])
        
        # Create variations
        words = template.split()
        num_words = len(words) + np.random.randint(-2, 8)
        if num_words < 5:
            num_words = 5
            
        # Select random words with replacement
        selected_words = np.random.choice(words, size=num_words, replace=True)
        
        # Add some random transition words
        transitions = ["and", "also", "including", "with", "because", "due to", "however"]
        for i in range(2, len(selected_words), 5):
            if i < len(selected_words):
                selected_words[i] = np.random.choice(transitions)
        
        document = " ".join(selected_words)
        documents.append(document)
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": documents,
        "actual_topic": doc_topics
    })
    
    print(f"Generated {len(df)} documents across {len(topic_names)} topics")
    return df

def main():
    """Generate a sample topic modeling report."""
    print("Generating sample topic modeling report...")
    
    # Generate sample data
    df = generate_sample_data(n_samples=100)
    
    # Initialize topic modeler
    modeler = MenoTopicModeler()
    
    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = modeler.preprocess(df, text_column="text")
    
    # Generate embeddings
    print("Generating document embeddings...")
    modeler.embed_documents()
    
    # Discover topics (unsupervised)
    print("Discovering topics using clustering...")
    unsupervised_results = modeler.discover_topics(
        method="embedding_cluster", 
        num_topics=4
    )
    
    # Create visualization
    print("Creating UMAP visualization...")
    fig = modeler.visualize_embeddings(return_figure=True)
    fig.write_html(SAMPLE_DIR / "embedding_visualization.html")
    
    # Generate distribution chart
    print("Creating topic distribution chart...")
    fig = modeler.visualize_topic_distribution(return_figure=True)
    fig.write_html(SAMPLE_DIR / "topic_distribution.html")
    
    # Generate HTML report
    print("Generating comprehensive HTML report...")
    report_path = modeler.generate_report(
        output_path=SAMPLE_DIR / "sample_topic_report.html",
        include_interactive=True
    )
    
    print(f"Sample report generated at {report_path}")
    
    # Supervised approach
    print("\nPerforming supervised topic matching...")
    
    predefined_topics = [
        "Customer Service Issues",
        "Claims Processing Problems",
        "Billing and Payment Concerns",
        "Policy Coverage Questions"
    ]
    
    topic_descriptions = [
        "Issues with customer service representatives, wait times, and support quality",
        "Problems with claims being processed, delayed, denied, or mishandled",
        "Billing errors, incorrect charges, payment issues, and fee disputes",
        "Questions about policy coverage, limits, exclusions, and terms"
    ]
    
    supervised_results = modeler.match_topics(
        topics=predefined_topics,
        descriptions=topic_descriptions,
        threshold=0.2,
    )
    
    print("Creating supervised topic visualization...")
    fig = modeler.visualize_embeddings(return_figure=True)
    fig.write_html(SAMPLE_DIR / "supervised_visualization.html")
    
    # Generate supervised report
    print("Generating supervised topic report...")
    report_path = modeler.generate_report(
        output_path=SAMPLE_DIR / "supervised_topic_report.html",
        include_interactive=True
    )
    
    print(f"Supervised report generated at {report_path}")
    print("\nAll sample reports generated successfully!")

if __name__ == "__main__":
    main()