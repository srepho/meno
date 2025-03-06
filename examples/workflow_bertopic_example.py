"""
Meno Workflow with BERTopic Integration Example

This example demonstrates how to integrate BERTopic with the Meno Workflow system
to combine the best of both approaches:
1. Interactive preprocessing with Meno Workflow
2. Advanced topic modeling with BERTopic
3. Visualization and reporting
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Meno and BERTopic components
from meno import MenoWorkflow
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

def load_data():
    """Load insurance dataset from Hugging Face."""
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

def run_workflow():
    """Run the integrated Meno Workflow with BERTopic."""
    
    # Load data
    df = load_data()
    
    # Initialize the Meno Workflow
    print("\nInitializing Meno Workflow...")
    workflow = MenoWorkflow()
    
    # Load data into workflow
    workflow.load_data(
        data=df,
        text_column="text",
        id_column="id"
    )
    
    # === STEP 1: Interactive Preprocessing ===
    
    # Generate acronym report
    print("\nDetecting and reporting acronyms...")
    acronym_report = workflow.generate_acronym_report(
        min_length=2,
        min_count=3,
        output_path=str(output_dir / "insurance_acronyms.html"),
        open_browser=False
    )
    
    # Define custom acronym mappings based on report
    acronym_mappings = {
        "PDS": "Product Disclosure Statement",
        "CTP": "Compulsory Third Party",
        "RSA": "Roadside Assistance",
        "CRM": "Customer Relationship Management",
        "SMS": "Short Message Service"
    }
    
    # Apply acronym expansions
    print("Expanding acronyms...")
    workflow.expand_acronyms(custom_mappings=acronym_mappings)
    
    # Generate spelling report
    print("\nDetecting and reporting potential misspellings...")
    spelling_report = workflow.generate_misspelling_report(
        min_length=5,
        min_count=2,
        output_path=str(output_dir / "insurance_misspellings.html"),
        open_browser=False
    )
    
    # Define custom spelling corrections based on report
    spelling_corrections = {
        "recieved": "received",
        "accross": "across",
        "reciept": "receipt",
        "occurence": "occurrence",
        "bussiness": "business"
    }
    
    # Apply spelling corrections
    print("Correcting spelling...")
    workflow.correct_spelling(custom_corrections=spelling_corrections)
    
    # Preprocess text data
    print("\nPreprocessing documents...")
    workflow.preprocess_documents(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        additional_stopwords=[
            "insurance", "policy", "claim", "insured", "insurer", "customer", 
            "premium", "please", "company", "dear", "sincerely", "writing", 
            "regards", "complaint", "email"
        ]
    )
    
    # Extract preprocessed data
    preprocessed_df = workflow.get_preprocessed_data()
    
    # === STEP 2: Advanced Topic Modeling with BERTopic ===
    
    print("\nConfiguring BERTopic components...")
    
    # Configure ClassTfidfTransformer
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True
    )
    
    # Configure representation models
    keybert_model = KeyBERTInspired()
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    
    # Create BERTopic model
    print("Creating BERTopic model...")
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        vectorizer_model=ctfidf_model,
        representation_model=[keybert_model, mmr_model],
        nr_topics=10,
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit the model
    print("\nFitting BERTopic model...")
    topics, probs = topic_model.fit_transform(preprocessed_df["processed_text"].tolist())
    
    # Add topics back to the DataFrame
    preprocessed_df["topic"] = [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics]
    preprocessed_df["topic_probability"] = probs
    
    # Update the workflow with BERTopic results
    print("\nUpdating workflow with BERTopic results...")
    workflow.set_topic_assignments(preprocessed_df[["topic", "topic_probability"]])
    
    # === STEP 3: Analysis and Visualization ===
    
    # Display topic information
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info[topic_info['Topic'] != -1])} topics")
    
    # Print top words for each topic
    print("\nTop words per topic:")
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic_id)
            words = [word for word, _ in topic_words[:5]]
            print(f"Topic {topic_id}: {', '.join(words)}")
    
    # Generate BERTopic visualizations
    print("\nGenerating BERTopic visualizations...")
    
    # Topic similarity
    topic_model.visualize_topics().write_html(str(output_dir / "bertopic_similarity.html"))
    
    # Topic hierarchy
    topic_model.visualize_hierarchy().write_html(str(output_dir / "bertopic_hierarchy.html"))
    
    # Topic barchart
    topic_model.visualize_barchart(top_n_topics=10).write_html(str(output_dir / "bertopic_barchart.html"))
    
    # === STEP 4: Meno Workflow Visualizations and Reporting ===
    
    print("\nGenerating Meno Workflow visualizations...")
    
    # Topic embedding visualization
    workflow.visualize_topics(plot_type="embeddings").write_html(str(output_dir / "topic_embeddings.html"))
    
    # Topic distribution visualization
    workflow.visualize_topics(plot_type="distribution").write_html(str(output_dir / "topic_distribution.html"))
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = workflow.generate_comprehensive_report(
        output_path=str(output_dir / "bertopic_integrated_report.html"),
        title="BERTopic Integrated Workflow Results",
        include_interactive=True,
        include_raw_data=True,
        open_browser=False
    )
    
    print(f"\nWorkflow complete! All outputs saved to {output_dir}")
    print(f"Comprehensive report: {report_path}")
    
    # List all generated files
    print("\nGenerated files:")
    for file in output_dir.glob("*.html"):
        print(f"- {file.name}")

if __name__ == "__main__":
    run_workflow()