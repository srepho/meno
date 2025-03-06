"""
Simple example showing BERTopic usage with minimal configuration.
"""

import pandas as pd
from datasets import load_dataset
from bertopic import BERTopic
import matplotlib.pyplot as plt

def main():
    """Run a simplified BERTopic example with minimal configuration."""
    
    print("Loading insurance dataset from Hugging Face...")
    dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
    df = pd.DataFrame({
        "text": dataset["train"]["original_text"],
        "id": dataset["train"]["id"]
    })
    print(f"Loaded {len(df)} documents")
    
    # Remove very short documents
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    
    print("\nInitializing and fitting BERTopic model...")
    # Create a BERTopic model with a single line of configuration
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",  # Small, fast model
        min_topic_size=10,                   # Minimum documents per topic
        nr_topics=12,                        # Target number of topics
        verbose=True
    )
    
    # Fit the model in a single step
    topics, probabilities = topic_model.fit_transform(df["text"].tolist())
    
    # Get information about discovered topics
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info[topic_info['Topic'] != -1])} topics")
    
    # Print top words for each topic
    print("\nTop words per topic:")
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic_id)
            words = [word for word, _ in topic_words[:5]]
            print(f"Topic {topic_id}: {', '.join(words)}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Topic word scores
    topic_model.visualize_barchart(top_n_topics=10).write_html("simple_bertopic_barchart.html")
    print("- Generated topic barchart: simple_bertopic_barchart.html")
    
    # 2. Topic similarity
    topic_model.visualize_topics().write_html("simple_bertopic_similarity.html")
    print("- Generated topic similarity visualization: simple_bertopic_similarity.html")
    
    # 3. Topic hierarchy
    topic_model.visualize_hierarchy().write_html("simple_bertopic_hierarchy.html")
    print("- Generated topic hierarchy: simple_bertopic_hierarchy.html")
    
    # Print a sample document from the largest topic
    largest_topic = topic_info.iloc[1]["Topic"]  # Skip -1 (outliers)
    topic_docs = df.iloc[[i for i, t in enumerate(topics) if t == largest_topic]]
    
    if not topic_docs.empty:
        print(f"\nSample document from Topic {largest_topic}:")
        print(topic_docs.iloc[0]["text"][:300] + "...")
    
    print("\nSimple BERTopic example completed successfully.")

if __name__ == "__main__":
    main()