"""
Generate a minimal sample report for documentation purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import meno
import sys
sys.path.append(str(Path(__file__).parent.parent))
from meno.meno import MenoTopicModeler

# Create output directory
SAMPLE_DIR = Path(__file__).parent / "sample_reports"
SAMPLE_DIR.mkdir(exist_ok=True)

def main():
    """Generate a minimal sample topic modeling report."""
    print("Generating sample topic modeling report...")
    
    # Sample data
    data = [
        "Customer service was poor and the representative was unhelpful.",
        "I had to wait a long time to speak with customer service.",
        "The staff was rude and the service experience was terrible.",
        "My claim was denied due to policy exclusions.",
        "The claim process is too slow and payment was delayed.",
        "There is too much documentation required for the claims process.",
        "I was overcharged and the bill had an incorrect amount.",
        "My automatic payment failed and I was charged twice.",
        "The bill statement is confusing with unclear charges.",
        "The coverage is inadequate with insufficient protection.",
        "I was misled about coverage by the salesperson.",
        "Coverage changed without notice and policy terms were modified."
    ]
    
    # Create dataframe
    df = pd.DataFrame({"text": data})
    
    # Initialize topic modeler
    modeler = MenoTopicModeler()
    
    # Preprocess documents
    processed_docs = modeler.preprocess(df, text_column="text")
    
    # Generate embeddings and discover topics
    modeler.embed_documents()
    topics_df = modeler.discover_topics(
        # Use lightweight model with duplicate topic detection enabled
        modeling_approach="lightweight",
        num_topics=10,  # Start with more topics, will be automatically reduced
        min_topic_similarity_threshold=0.6,
        auto_reduce_topics=True
    )
    
    # Create visualizations
    umap_fig = modeler.visualize_embeddings(return_figure=True)
    umap_fig.write_html(SAMPLE_DIR / "sample_embedding_viz.html")
    
    dist_fig = modeler.visualize_topic_distribution(return_figure=True)
    dist_fig.write_html(SAMPLE_DIR / "sample_distribution.html")
    
    print("Sample visualizations saved to examples/sample_reports/")

if __name__ == "__main__":
    main()