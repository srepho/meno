# Topic Feedback System - Interactive Topic Refinement
#
# This example demonstrates how to use the topic feedback system to
# refine topic assignments based on human input.

import pandas as pd
import matplotlib.pyplot as plt
from meno import MenoTopicModeler
from meno.active_learning.simple_feedback import TopicFeedbackManager
from meno.visualization.enhanced_viz.feedback_viz import (
    plot_feedback_impact,
    plot_topic_feedback_distribution,
    create_feedback_comparison_dashboard
)

# Sample data (in a real application, load from a file)
documents = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Topic modeling is a type of statistical modeling for discovering abstract topics in document collections.",
    "Natural language processing (NLP) is a field of artificial intelligence focused on interactions between computers and human language.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Unsupervised learning is where the algorithm is given data without explicit instructions on what to do with it.",
    "Supervised learning algorithms build a model based on labeled training data.",
    "Clustering is the task of dividing data points into groups based on similarity.",
    "Classification in machine learning is the task of predicting which category an observation belongs to.",
    "Regression analysis is used to estimate the relationships among variables.",
    "Dimensionality reduction is the process of reducing the number of variables under consideration.",
    "NLP involves natural language understanding and generation through computational methods.",
    "Word embeddings are vector representations of words that capture semantic relationships.",
    "Transfer learning involves applying knowledge from one task to improve performance on another task.",
    "Ensemble methods combine multiple machine learning models to improve performance.",
    "Feature engineering is the process of transforming raw data into features suitable for machine learning."
]

# Create a DataFrame
df = pd.DataFrame({"text": documents})

print("Step 1: Initial Topic Modeling")
print("-" * 50)

# Initialize modeler
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="text")

# Discover topics (using automatic detection)
modeler.discover_topics(method="embedding_cluster", auto_detect_topics=True)

# View initial topic assignments
topic_info = modeler.get_topic_info()
print("Initial topic distribution:")
print(topic_info)

# Get document-topic assignments
doc_topics = modeler.get_document_topics()
print("\nSample document-topic assignments:")
print(doc_topics.head())

print("\nStep 2: Setup Feedback System")
print("-" * 50)

# Create feedback manager
feedback_manager = TopicFeedbackManager(modeler)

# Set up topic descriptions for better human interpretation
topic_descriptions = {}
for topic_id in topic_info["Topic"].unique():
    if topic_id >= 0:  # Skip outlier topic (-1)
        words = modeler.get_topic_words(topic_id)
        topic_descriptions[f"Topic_{topic_id}"] = f"Words: {', '.join(words[:5])}"

print("Topic descriptions:")
for topic, desc in topic_descriptions.items():
    print(f"- {topic}: {desc}")

# Setup feedback system
feedback_system = feedback_manager.setup_feedback(
    n_samples=5,               # Number of documents to review
    uncertainty_ratio=0.7,     # Focus on uncertain documents
    topic_descriptions=topic_descriptions
)

print("\nStep 3: Simulate User Feedback")
print("-" * 50)
print("In an interactive environment, we would use the start_review() method.")
print("For this example, we'll simulate feedback programmatically.")

# Store original topic assignments before feedback
original_topics = doc_topics["Topic"].tolist()

# Simulate feedback (in a real scenario, this comes from user interaction)
# For demonstration purposes, we'll make some arbitrary changes:
# - Reassign document 0 to topic 1 (if topic 1 exists)
# - Mark document 1 as an outlier
# - Confirm document 2's existing assignment

# Get available topics
available_topics = [f"Topic_{t}" for t in topic_info["Topic"].unique() if t >= 0]

# Simulated feedback
feedback = []

# Document 0: Reassign to a different topic if possible
if len(available_topics) > 1:
    current_topic = doc_topics.iloc[0]["Topic"]
    new_topic = [t for t in available_topics if t != current_topic][0]
    feedback.append({
        "document_idx": 0,
        "document": documents[0],
        "original_topic": current_topic,
        "feedback": "reassign",
        "new_topic": new_topic
    })
    print(f"Simulated feedback: Reassign document 0 from {current_topic} to {new_topic}")

# Document 1: Mark as outlier
feedback.append({
    "document_idx": 1,
    "document": documents[1],
    "original_topic": doc_topics.iloc[1]["Topic"],
    "feedback": "outlier",
    "new_topic": "Outlier"
})
print(f"Simulated feedback: Mark document 1 as an outlier")

# Document 2: Confirm existing assignment
feedback.append({
    "document_idx": 2,
    "document": documents[2],
    "original_topic": doc_topics.iloc[2]["Topic"],
    "feedback": "confirm",
    "new_topic": doc_topics.iloc[2]["Topic"]
})
print(f"Simulated feedback: Confirm document 2's assignment to {doc_topics.iloc[2]['Topic']}")

# Manually set feedback instead of using the interactive interface
feedback_system.feedback = feedback

print("\nStep 4: Apply Feedback")
print("-" * 50)

# Apply updates based on feedback
feedback_system.apply_updates()

# Get the updated model
updated_modeler = feedback_manager.get_updated_model()

# View updated topic information
updated_topic_info = updated_modeler.get_topic_info()
print("Updated topic distribution:")
print(updated_topic_info)

# Get updated document-topic assignments
updated_doc_topics = updated_modeler.get_document_topics()
print("\nUpdated document-topic assignments for feedback documents:")
for idx in [0, 1, 2]:
    old_topic = doc_topics.iloc[idx]["Topic"]
    new_topic = updated_doc_topics[updated_doc_topics["Document"] == documents[idx]]["Topic"].values[0]
    print(f"Document {idx}: {old_topic} -> {new_topic}")

print("\nStep 5: Visualize Feedback Impact")
print("-" * 50)

try:
    # Visualize the impact of feedback
    fig = plot_feedback_impact(feedback_manager)
    
    # In a script context, we save the figure instead of displaying it
    plt.figure(fig.number)
    plt.savefig("feedback_impact.png")
    print("Saved feedback impact visualization to feedback_impact.png")
    
    # Topic-specific changes
    current_topics = updated_doc_topics["Topic"].tolist()
    fig2 = plot_topic_feedback_distribution(
        updated_modeler,
        documents,
        original_topics,
        current_topics,
        show_wordclouds=True
    )
    plt.figure(fig2.number)
    plt.savefig("topic_distribution_changes.png")
    print("Saved topic distribution changes to topic_distribution_changes.png")
except Exception as e:
    print(f"Visualization could not be generated: {e}")

print("\nStep 6: Create Interactive Dashboard (requires Dash)")
print("-" * 50)

try:
    # Create a dashboard application
    app = create_feedback_comparison_dashboard(
        before_model=modeler,         # Original model
        after_model=updated_modeler,  # Updated model
        documents=documents,
        title="Feedback Impact Analysis"
    )
    
    print("Dashboard created. In a real application, you would launch it with:")
    print("app.run_server(debug=True)")
    
    # Note: We don't actually run the server in this example script
except Exception as e:
    print(f"Dashboard could not be created: {e}")
    print("Note: This requires dash to be installed")

print("\nStep 7: Export Feedback")
print("-" * 50)

# Export feedback to CSV
try:
    export_path = "topic_feedback.csv"
    feedback_system.export_to_csv(export_path)
    print(f"Feedback exported to {export_path}")
    
    # Show exported data
    feedback_df = pd.read_csv(export_path)
    print("\nExported feedback data:")
    print(feedback_df.head())
except Exception as e:
    print(f"Could not export feedback: {e}")

print("\nConclusion:")
print("-" * 50)
print("This example demonstrated how to:")
print("1. Set up a topic feedback system")
print("2. Provide feedback on topic assignments")
print("3. Apply changes based on feedback")
print("4. Visualize the impact of feedback")
print("5. Export feedback for sharing or record-keeping")
print("\nThe interactive feedback system is best used in a Jupyter notebook environment.")
print("See examples/notebooks/topic_feedback.ipynb for the interactive version.")