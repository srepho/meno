"""
External Topic Integration Example

This example demonstrates how to:
1. Preprocess data using Meno's workflow
2. Extract the preprocessed data using get_preprocessed_data() 
3. Apply an external topic modeling algorithm
4. Integrate the results back into Meno using set_topic_assignments()
5. Leverage Meno's visualization and reporting capabilities

Two examples are shown:
- Using scikit-learn's NMF for topic modeling
- Using BERTopic for more advanced topic modeling
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Add parent directory to path to import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Meno components
from meno import MenoWorkflow

# Flag to determine which external model to use
USE_BERTOPIC = True  # Set to False to use NMF instead

def load_data():
    """Load sample dataset from Hugging Face."""
    print("Loading insurance dataset from Hugging Face...")
    dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
    
    # Convert to DataFrame with properly mapped columns
    df = pd.DataFrame({
        "text": dataset["train"]["original_text"],
        "id": dataset["train"]["id"]
    })
    
    # Take a sample for faster processing
    df = df.sample(n=200, random_state=42)
    print(f"Loaded {len(df)} documents")
    
    return df

def topic_modeling_with_nmf(documents, num_topics=10):
    """Apply scikit-learn's NMF for topic modeling.
    
    Parameters
    ----------
    documents : List[str]
        List of preprocessed documents
    num_topics : int
        Number of topics to extract
        
    Returns
    -------
    pd.DataFrame
        DataFrame with topic assignments and probabilities
    """
    print(f"\nPerforming NMF topic modeling with {num_topics} topics...")
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.85
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Apply NMF
    nmf_model = NMF(
        n_components=num_topics,
        random_state=42,
        max_iter=200
    )
    doc_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
    
    # Get the dominant topic for each document
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    
    # Create topic names based on top terms
    topic_names = {}
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_idx = topic.argsort()[:-10 - 1:-1]
        top_terms = [feature_names[i] for i in top_features_idx]
        topic_names[topic_idx] = f"Topic_{topic_idx}: {', '.join(top_terms[:3])}"
    
    # Create topic assignments DataFrame
    topic_assignments = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"Topic_{i}" for i in range(num_topics)]
    )
    
    # Add dominant topic and probability
    topic_assignments["topic"] = [topic_names[t] for t in dominant_topics]
    topic_assignments["topic_probability"] = np.max(doc_topic_matrix, axis=1)
    
    # Print a summary of discovered topics
    print("\nDiscovered topics:")
    for topic_idx, name in topic_names.items():
        count = np.sum(dominant_topics == topic_idx)
        print(f"- {name} ({count} documents)")
    
    return topic_assignments

def topic_modeling_with_bertopic(documents, num_topics=10):
    """Apply BERTopic for advanced topic modeling.
    
    Parameters
    ----------
    documents : List[str]
        List of preprocessed documents
    num_topics : int
        Number of topics to extract
        
    Returns
    -------
    pd.DataFrame
        DataFrame with topic assignments and probabilities
    """
    # Import BERTopic (only if needed, to avoid dependency if not used)
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    
    print(f"\nPerforming BERTopic modeling with target of {num_topics} topics...")
    
    # Configure BERTopic model
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",  # Lightweight embedding model
        nr_topics=num_topics,
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(documents)
    
    # Get topic info for reporting
    topic_info = topic_model.get_topic_info()
    actual_num_topics = len(topic_info[topic_info['Topic'] != -1])
    print(f"\nActually discovered {actual_num_topics} topics")
    
    # Convert to topic names with descriptive labels
    topic_names = {}
    for row in topic_info.itertuples():
        topic_id = row.Topic
        if topic_id == -1:
            topic_names[topic_id] = "Outlier"
        else:
            # Get representative terms
            terms = [term for term, _ in topic_model.get_topic(topic_id)][:5]
            topic_names[topic_id] = f"Topic_{topic_id}: {', '.join(terms[:3])}"
    
    # Create topic assignments DataFrame
    all_topics = sorted(list(set(topics)))
    topic_probs_df = pd.DataFrame(
        # Create a matrix of probabilities - one column per topic
        0,  # Initialize with zeros
        index=range(len(documents)),
        columns=[f"Topic_{i}" if i >= 0 else "Outlier" for i in all_topics]
    )
    
    # Fill in probabilities where available
    for i, (topic, prob) in enumerate(zip(topics, probs)):
        topic_name = f"Topic_{topic}" if topic >= 0 else "Outlier"
        topic_probs_df.loc[i, topic_name] = prob
    
    # Add topic assignment columns
    topic_probs_df["topic"] = [topic_names[t] for t in topics]
    topic_probs_df["topic_probability"] = probs
    
    # Print a summary of discovered topics
    print("\nDiscovered topics:")
    for topic_id, name in topic_names.items():
        if topic_id != -1:  # Skip outlier topic in summary
            count = sum(1 for t in topics if t == topic_id)
            print(f"- {name} ({count} documents)")
    
    return topic_probs_df

def main():
    """Run the example workflow."""
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load data
    df = load_data()
    
    # 2. Initialize Meno workflow
    print("\nInitializing Meno workflow...")
    workflow = MenoWorkflow()
    
    # 3. Load data into workflow
    workflow.load_data(
        data=df,
        text_column="text",
        id_column="id"
    )
    
    # 4. Preprocess the data with Meno's pipeline
    print("\nPreprocessing documents with Meno...")
    workflow.preprocess_documents(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        additional_stopwords=[
            "insurance", "policy", "claim", "insured", "insurer", 
            "customer", "premium", "please", "company", "dear", 
            "sincerely", "regards", "complaint", "email"
        ]
    )
    
    # 5. Get the preprocessed data
    preprocessed_df = workflow.get_preprocessed_data()
    print(f"Retrieved {len(preprocessed_df)} preprocessed documents")
    
    # 6. Apply external topic modeling (either BERTopic or NMF)
    if USE_BERTOPIC:
        # Use BERTopic for topic modeling
        try:
            topic_assignments = topic_modeling_with_bertopic(
                preprocessed_df["processed_text"].tolist(),
                num_topics=8
            )
        except ImportError:
            print("\nBERTopic not installed. Please install with:")
            print("pip install bertopic sentence-transformers")
            print("\nFalling back to NMF...")
            topic_assignments = topic_modeling_with_nmf(
                preprocessed_df["processed_text"].tolist(),
                num_topics=8
            )
    else:
        # Use NMF for topic modeling
        topic_assignments = topic_modeling_with_nmf(
            preprocessed_df["processed_text"].tolist(),
            num_topics=8
        )
    
    # 7. Integrate external topic modeling results back into Meno
    print("\nIntegrating external topic modeling results back into Meno workflow...")
    workflow.set_topic_assignments(topic_assignments[["topic", "topic_probability"]])
    
    # 8. Use Meno's visualization capabilities
    print("\nGenerating visualizations...")
    
    # Create embedding visualization
    try:
        embedding_viz = workflow.visualize_topics(plot_type="embeddings")
        embedding_viz.write_html(str(output_dir / "external_topic_embeddings.html"))
        print(f"Saved embedding visualization to {output_dir}/external_topic_embeddings.html")
    except Exception as e:
        print(f"Could not generate embedding visualization: {e}")
    
    # Create topic distribution visualization
    try:
        dist_viz = workflow.visualize_topics(plot_type="distribution")
        dist_viz.write_html(str(output_dir / "external_topic_distribution.html"))
        print(f"Saved topic distribution visualization to {output_dir}/external_topic_distribution.html")
    except Exception as e:
        print(f"Could not generate distribution visualization: {e}")
    
    # 9. Generate comprehensive HTML report
    print("\nGenerating comprehensive report...")
    report_path = workflow.generate_comprehensive_report(
        output_path=str(output_dir / "external_topic_modeling_report.html"),
        title=f"External Topic Modeling with {'BERTopic' if USE_BERTOPIC else 'NMF'}",
        include_interactive=True,
        include_raw_data=True,
        open_browser=True
    )
    
    print(f"\nWorkflow complete! Report saved to {report_path}")
    print("\nThis example demonstrated how to:")
    print("1. Preprocess data using Meno's workflow")
    print("2. Extract preprocessed data for external processing")
    print("3. Apply an external topic modeling algorithm")
    print("4. Integrate results back into Meno")
    print("5. Generate visualizations and reports using Meno's capabilities")

if __name__ == "__main__":
    main()