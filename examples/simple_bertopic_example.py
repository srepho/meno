"""
Simple example showing BERTopic usage with ClassTfidfTransformer.
"""

import pandas as pd
from datasets import load_dataset
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import matplotlib.pyplot as plt

def main():
    """Run a simplified BERTopic example with ClassTfidfTransformer."""
    
    print("Loading insurance dataset from Hugging Face...")
    dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
    df = pd.DataFrame({
        "text": dataset["train"]["original_text"],
        "id": dataset["train"]["id"]
    })
    print(f"Loaded {len(df)} documents")
    
    # Remove very short documents
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    
    print("\nInitializing custom vectorizer...")
    # Create a ClassTfidfTransformer with custom settings
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,  # Reduce impact of frequent words
        bm25_weighting=True,         # Use BM25 weighting for better results
        seed_topic_list=None,        # Optional seeding with domain-specific terms
        min_df=5                     # Minimum document frequency for terms
    )
    
    print("\nInitializing and fitting BERTopic model...")
    # Create a BERTopic model with the custom vectorizer
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",  # Small, fast model
        vectorizer_model=ctfidf_model,       # Use our custom vectorizer
        min_topic_size=10,                   # Minimum documents per topic
        nr_topics=12,                        # Target number of topics
        verbose=True
    )
    
    # Fit the model in a single step
    topics, probabilities = topic_model.fit_transform(df["text"].tolist())
    
    # Get information about discovered topics
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info[topic_info['Topic'] != -1])} topics")
    
    # Print top words for each topic with scores (showing ClassTfidfTransformer impact)
    print("\nTop words per topic with c-TF-IDF scores:")
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic_id)
            words_with_scores = [(word, f"{score:.3f}") for word, score in topic_words[:5]]
            words_formatted = [f"{word} ({score})" for word, score in words_with_scores]
            print(f"Topic {topic_id}: {', '.join(words_formatted)}")
    
    # You can also analyze how the c-TF-IDF model affects the topic representations
    print("\nDemonstrating c-TF-IDF impact on topics...")
    # Get the fitted c-TF-IDF model from BERTopic
    ctfidf_model = topic_model.vectorizer_model
    
    # Get the feature names (words) from the c-TF-IDF model
    if hasattr(ctfidf_model, 'feature_names_'):
        # For some versions of BERTopic
        feature_names = ctfidf_model.feature_names_
        if feature_names is not None:
            print(f"Number of unique terms in the c-TF-IDF vocabulary: {len(feature_names)}")
    
    # Get the most distinctive terms for a specific topic
    topic_id_to_analyze = 0  # Choose a topic to analyze
    if topic_id_to_analyze in topic_model.get_topics():
        topic_terms = topic_model.get_topic(topic_id_to_analyze)
        print(f"\nAnalysis of terms for Topic {topic_id_to_analyze}:")
        print("Term                  c-TF-IDF Score    Description")
        print("----------------------------------------------------")
        for term, score in topic_terms[:10]:
            print(f"{term:<20} {score:.5f}         {'High frequency in this topic, low in others' if score > 0.3 else 'Moderately distinctive term'}")
    
    # Print information about the BM25 weighting if it was used
    print("\nc-TF-IDF configuration:")
    print(f"- BM25 weighting: {ctfidf_model.bm25_weighting}")
    print(f"- Reduce frequent words: {ctfidf_model.reduce_frequent_words}")
    
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