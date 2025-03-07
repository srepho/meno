"""Example script demonstrating the simple feedback system for topic modeling.

This script shows how to use the SimpleFeedback class to incorporate
user feedback into topic models in a Jupyter notebook environment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Meno components
from meno import MenoTopicModeler
from meno.active_learning.simple_feedback import SimpleFeedback, TopicFeedbackManager


def create_sample_data():
    """Create sample data for demonstration."""
    # Create sample data with 3 topic categories
    np.random.seed(42)
    
    # Technology documents
    tech_docs = [
        "Machine learning algorithms can analyze large datasets to identify patterns.",
        "Neural networks are computing systems inspired by biological brains.",
        "Deep learning uses multiple layers to progressively extract higher level features.",
        "Artificial intelligence is intelligence demonstrated by machines.",
        "Data science combines domain expertise, programming, and math to extract insights."
    ]
    
    # Healthcare documents
    health_docs = [
        "Medical imaging techniques like MRI help diagnose conditions non-invasively.",
        "Electronic health records store patient information digitally.",
        "Telemedicine allows healthcare providers to consult with patients remotely.",
        "Preventive medicine focuses on maintaining health and preventing disease.",
        "Personalized medicine tailors treatments based on individual genetic profiles."
    ]
    
    # Environmental documents
    env_docs = [
        "Climate change refers to long-term shifts in global weather patterns.",
        "Renewable energy sources like solar and wind power help reduce emissions.",
        "Conservation efforts focus on protecting biodiversity and natural habitats.",
        "Sustainable development balances economic growth with environmental protection.",
        "Electric vehicles reduce reliance on fossil fuels and lower carbon emissions."
    ]
    
    # Deliberately add some documents that could be in multiple categories
    mixed_docs = [
        "AI is being used to analyze medical images for more accurate diagnoses.",  # tech + health
        "Machine learning helps predict climate patterns and environmental changes.",  # tech + env
        "Electronic health records are stored in data centers that consume significant energy.",  # health + env
        "Wearable technology monitors health metrics and sends data to healthcare providers.",  # tech + health
        "Smart grid technology optimizes energy distribution and reduces waste."  # tech + env
    ]
    
    # Combine all documents
    all_docs = tech_docs + health_docs + env_docs + mixed_docs
    
    # Create a true category label for each document (for evaluation)
    true_categories = (["Technology"] * 5 + ["Healthcare"] * 5 + 
                      ["Environment"] * 5 + ["Mixed"] * 5)
    
    # Create a dataframe
    df = pd.DataFrame({
        "text": all_docs,
        "true_category": true_categories
    })
    
    print(f"Created sample dataset with {len(df)} documents")
    return df


def run_topic_modeling(df):
    """Run topic modeling on the sample data."""
    # Initialize topic modeler
    modeler = MenoTopicModeler()
    
    # Preprocess the data
    modeler.preprocess(df, text_column="text")
    
    # Discover topics (we know there are 3 main topics)
    topic_df = modeler.discover_topics(
        method="embedding_cluster",
        num_topics=3,
        random_state=42
    )
    
    # Print topic information
    topic_info = modeler.get_topic_info()
    print("\nDiscovered Topics:")
    for _, row in topic_info.iterrows():
        print(f"Topic {row['Topic']}: {row['Name']}")
    
    # Show document information
    doc_info = modeler.get_document_info()
    print("\nSample Document Assignments:")
    for i in range(5):  # Show first 5 documents
        print(f"Document {i}: {doc_info.iloc[i]['Topic']} - {df.iloc[i]['text'][:50]}...")
    
    return modeler


def evaluate_topics(modeler, df, before_feedback=True):
    """Evaluate topic coherence before or after feedback."""
    # Get document topic assignments
    doc_info = modeler.get_document_info()
    
    # Count documents by topic and true category
    topic_category_counts = pd.crosstab(
        doc_info["Topic"],
        df["true_category"],
        margins=True,
        margins_name="Total"
    )
    
    # Display the counts
    print(f"\nTopic-Category Distribution {'(Before Feedback)' if before_feedback else '(After Feedback)'}:")
    print(topic_category_counts)
    
    # Calculate metrics: what percentage of each topic comes from each true category
    topic_composition = topic_category_counts.div(topic_category_counts["Total"], axis=0) * 100
    topic_composition = topic_composition.drop("Total", axis=1)
    
    # Plot topic composition
    plt.figure(figsize=(12, 6))
    topic_composition.drop("Total", axis=0).plot(
        kind="bar",
        title=f"Topic Composition {'(Before Feedback)' if before_feedback else '(After Feedback)'}"
    )
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="True Category")
    plt.tight_layout()
    plt.show()


def simple_feedback_example(modeler, df):
    """Demonstrate the simple feedback system."""
    # Get document and topic information
    documents = modeler.get_preprocessed_documents()
    topic_info = modeler.get_topic_info()
    doc_info = modeler.get_document_info()
    
    # Create topic descriptions for better understanding
    topic_descriptions = [
        "Documents related to technology, computing, and data science",
        "Documents related to healthcare, medicine, and patient care",
        "Documents related to environment, climate, and sustainability"
    ]
    
    # Create simple feedback system
    feedback = SimpleFeedback(
        documents=documents,
        topics=doc_info["Topic"].tolist(),
        topic_names=topic_info["Name"].tolist(),
        topic_descriptions=topic_descriptions,
        callback=lambda updated_topics: print(f"Updated {sum(1 for i, t in enumerate(updated_topics) if t != doc_info['Topic'].tolist()[i])} topics")
    )
    
    # Display the topics for reference
    feedback.display_topics()
    
    # Get documents to review - include some potential misclassifications
    # In a real notebook, this would be interactive
    print("\nIn a Jupyter notebook, you would now see an interactive interface")
    print("for providing feedback on document topic assignments.")
    print("\nThis example script simulates providing feedback by identifying")
    print("documents that are likely to be misclassified (mixed category docs).")
    
    # Select the mixed documents (these may be misclassified)
    mixed_indices = df[df["true_category"] == "Mixed"].index.tolist()
    feedback.set_documents_to_review(mixed_indices)
    
    # Start the review (in a notebook, this would be interactive)
    # Simulate feedback by directly updating the topics
    for idx in mixed_indices:
        text = df.iloc[idx]["text"]
        print(f"\nDocument: {text}")
        print(f"Original topic: {feedback.df.at[idx, 'original_topic']}")
        
        # Simulate user feedback based on document content
        if "medical" in text.lower() or "health" in text.lower():
            new_topic = topic_info["Name"][1]  # Healthcare topic
            print(f"Feedback assigns to: {new_topic}")
            feedback.df.at[idx, "current_topic"] = new_topic
            feedback.feedback[idx] = new_topic
        elif "energy" in text.lower() or "climate" in text.lower():
            new_topic = topic_info["Name"][2]  # Environment topic 
            print(f"Feedback assigns to: {new_topic}")
            feedback.df.at[idx, "current_topic"] = new_topic
            feedback.feedback[idx] = new_topic
        elif "AI" in text or "machine learning" in text.lower():
            new_topic = topic_info["Name"][0]  # Technology topic
            print(f"Feedback assigns to: {new_topic}")
            feedback.df.at[idx, "current_topic"] = new_topic
            feedback.feedback[idx] = new_topic
    
    # Display feedback summary
    print("\nFeedback Summary:")
    feedback.display_summary()
    
    # Update the model with feedback
    updated_topics = feedback.get_updated_topics()
    
    # In a real application, we would update the model
    # Here we'll manually update the document info
    for i, topic in enumerate(updated_topics):
        doc_info.at[i, "Topic"] = topic
    
    # Return the updated modeler
    return modeler


def feedback_manager_example(df):
    """Demonstrate the TopicFeedbackManager for easier integration."""
    # Run topic modeling
    modeler = MenoTopicModeler()
    modeler.preprocess(df, text_column="text")
    modeler.discover_topics(method="embedding_cluster", num_topics=3)
    
    # Create a feedback manager
    feedback_manager = TopicFeedbackManager(modeler)
    
    # Set up feedback system
    topic_descriptions = [
        "Documents related to technology, computing, and data science",
        "Documents related to healthcare, medicine, and patient care",
        "Documents related to environment, climate, and sustainability"
    ]
    
    feedback_system = feedback_manager.setup_feedback(
        n_samples=10,  # Review 10 documents
        uncertainty_ratio=0.7,  # 70% uncertain documents, 30% diverse documents
        topic_descriptions=topic_descriptions
    )
    
    # In a notebook, this would launch the interactive interface
    print("\nWith TopicFeedbackManager, you would run:")
    print("feedback_manager.start_review()")
    print("\nThis would provide an interactive interface in a Jupyter notebook")
    print("for reviewing and correcting topic assignments.")
    
    # Simulate feedback (in a real notebook, this would be interactive)
    mixed_indices = df[df["true_category"] == "Mixed"].index.tolist()
    topic_names = modeler.get_topic_info()["Name"].tolist()
    
    for idx in mixed_indices[:2]:  # Just update a couple as an example
        text = df.iloc[idx]["text"]
        current_topic = feedback_system.df.at[idx, "original_topic"]
        
        print(f"\nDocument: {text}")
        print(f"Original topic: {current_topic}")
        
        # Simulate user feedback
        if "medical" in text.lower() or "health" in text.lower():
            new_topic = topic_names[1]  # Healthcare topic
            print(f"Feedback assigns to: {new_topic}")
            feedback_system.df.at[idx, "current_topic"] = new_topic
            feedback_system.feedback[idx] = new_topic
    
    # Display feedback summary
    print("\nFeedback Summary:")
    feedback_system.display_summary()
    
    # Apply updates to the model
    feedback_system.apply_updates()
    
    # Get the updated model
    updated_modeler = feedback_manager.get_updated_model()
    
    print("\nModel updated with feedback!")
    return updated_modeler


def main():
    """Run the example script."""
    print("=" * 80)
    print("MENO INTERACTIVE FEEDBACK EXAMPLE")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data()
    
    # Run topic modeling
    print("\n" + "=" * 50)
    print("Running initial topic modeling...")
    modeler = run_topic_modeling(df)
    
    # Evaluate topics before feedback
    print("\n" + "=" * 50)
    print("Evaluating initial topic assignments...")
    evaluate_topics(modeler, df, before_feedback=True)
    
    # Demonstrate simple feedback system
    print("\n" + "=" * 50)
    print("Demonstrating SimpleFeedback...")
    updated_modeler = simple_feedback_example(modeler, df)
    
    # Evaluate topics after feedback
    print("\n" + "=" * 50)
    print("Evaluating topic assignments after feedback...")
    evaluate_topics(updated_modeler, df, before_feedback=False)
    
    # Demonstrate the feedback manager
    print("\n" + "=" * 50)
    print("Demonstrating TopicFeedbackManager...")
    feedback_manager_example(df)
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nIn a Jupyter notebook, this would be interactive with UI elements")
    print("for selecting topics and providing feedback.")


if __name__ == "__main__":
    main()