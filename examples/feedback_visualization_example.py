"""
Feedback Visualization Example for Meno Topic Modeling Toolkit

This example demonstrates how to use the new feedback visualization components
to analyze and visualize the impact of feedback on topic models.
"""

import meno
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from IPython.display import display
import sys
import os

# Add the correct path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import feedback components
from meno import SimpleFeedback, TopicFeedbackManager
from meno import plot_feedback_impact, plot_topic_feedback_distribution
from meno import create_feedback_comparison_dashboard

# Load sample dataset (20 newsgroups)
print("Loading 20 newsgroups dataset...")
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=['alt.atheism', 'comp.graphics', 'sci.space'],
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Use a small subset for this example
sample_size = 500
indices = np.random.RandomState(42).choice(len(newsgroups.data), sample_size, replace=False)
documents = [newsgroups.data[i] for i in indices]
labels = [newsgroups.target[i] for i in indices]

# Print dataset info
print(f"Loaded {len(documents)} documents from {len(newsgroups.target_names)} categories")
print(f"Categories: {newsgroups.target_names}")

# Create a simple topic model
print("Creating topic model...")
model = meno.TFIDFTopicModel(n_topics=5)
model.fit(documents)

# Get document topics
doc_topics = model.get_document_topics()
topics = doc_topics["topic"].tolist()

print("Topics assigned:")
topic_counts = pd.Series(topics).value_counts().sort_index()
for topic, count in topic_counts.items():
    print(f"  Topic {topic}: {count} documents")

# Simulate feedback by changing some topics
print("\nSimulating user feedback...")
original_topics = topics.copy()
corrected_topics = topics.copy()

# Randomly change 10% of documents
np.random.seed(42)
n_changes = int(len(documents) * 0.10)
change_indices = np.random.choice(len(documents), n_changes, replace=False)

for idx in change_indices:
    current_topic = corrected_topics[idx]
    # Assign to a different topic
    available_topics = [t for t in range(model.n_topics) if t != current_topic]
    corrected_topics[idx] = np.random.choice(available_topics)

# Print changes
changed_count = sum(1 for a, b in zip(original_topics, corrected_topics) if a != b)
print(f"Changed {changed_count} documents ({changed_count/len(documents):.1%} of total)")

# Create a simulated feedback manager
class MockFeedbackManager:
    def __init__(self, documents, original_topics, corrected_topics):
        self.documents = documents
        self.original_topics = original_topics
        self.corrected_topics = corrected_topics
        
        # Create feedback sessions data
        self.feedback_sessions = []
        
        # Split changes into 3 sessions
        change_indices = [i for i, (a, b) in enumerate(zip(original_topics, corrected_topics)) if a != b]
        session_splits = np.array_split(change_indices, 3)
        
        for i, session_indices in enumerate(session_splits):
            feedback = []
            for idx in session_indices:
                feedback.append({
                    "document_index": idx,
                    "original_topic": original_topics[idx],
                    "corrected_topic": corrected_topics[idx]
                })
            
            session = {
                "session_id": i + 1,
                "timestamp": f"2025-03-0{i+1} 10:00:00",
                "feedback": feedback,
                "n_documents": len(session_indices),
                "n_changed": len(session_indices),
                "pct_changed": 100.0  # All documents in this mock changed
            }
            self.feedback_sessions.append(session)
        
        # Prepare topic counts
        self.original_topic_counts = {}
        self.current_topic_counts = {}
        
        for topic in range(model.n_topics):
            self.original_topic_counts[topic] = original_topics.count(topic)
            self.current_topic_counts[topic] = corrected_topics.count(topic)
            
        # Prepare transitions
        self.topic_transitions = []
        transition_counts = {}
        
        for idx in change_indices:
            from_topic = original_topics[idx]
            to_topic = corrected_topics[idx]
            key = (from_topic, to_topic)
            
            if key not in transition_counts:
                transition_counts[key] = 0
            transition_counts[key] += 1
        
        for (from_topic, to_topic), count in transition_counts.items():
            self.topic_transitions.append({
                "from_topic": from_topic,
                "to_topic": to_topic,
                "count": count
            })
    
    def get_feedback_summary(self):
        """Return summary of feedback sessions"""
        summaries = []
        for session in self.feedback_sessions:
            summary = {
                "session_id": session["session_id"],
                "timestamp": session["timestamp"],
                "documents_reviewed": session["n_documents"],
                "topics_changed": session["n_changed"],
                "percent_changed": session["pct_changed"],
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def evaluate_impact(self, detailed=False):
        """Evaluate impact of feedback"""
        # Calculate changes
        documents_changed = sum(1 for a, b in zip(self.original_topics, self.corrected_topics) if a != b)
        
        # Find changed indices
        changed_indices = [i for i, (a, b) in enumerate(zip(self.original_topics, self.corrected_topics)) if a != b]
        
        # Topic changes
        topic_changes = {}
        for topic in range(model.n_topics):
            orig_count = self.original_topics.count(topic)
            curr_count = self.corrected_topics.count(topic)
            topic_changes[topic] = curr_count - orig_count
        
        result = {
            "total_documents": len(self.documents),
            "documents_changed": documents_changed,
            "percent_changed": (documents_changed / len(self.documents)) * 100,
            "topic_changes": topic_changes,
            "original_topic_counts": self.original_topic_counts,
            "current_topic_counts": self.current_topic_counts,
            "topic_transitions": self.topic_transitions
        }
        
        if detailed:
            # Include detailed document changes
            detailed_changes = []
            for idx in changed_indices:
                detailed_changes.append({
                    "document_index": idx,
                    "document_text": self.documents[idx][:100] + "...",  # Truncate for brevity
                    "original_topic": self.original_topics[idx],
                    "current_topic": self.corrected_topics[idx],
                })
            result["detailed_changes"] = detailed_changes
        
        return result

# Create mock feedback manager
feedback_manager = MockFeedbackManager(documents, original_topics, corrected_topics)

# Print feedback sessions
print("\nFeedback sessions:")
print(feedback_manager.get_feedback_summary())

# Simple impact assessment
impact = feedback_manager.evaluate_impact()
print("\nFeedback impact:")
print(f"Total documents: {impact['total_documents']}")
print(f"Documents changed: {impact['documents_changed']} ({impact['percent_changed']:.1f}%)")

print("\nTopic changes:")
for topic, change in impact['topic_changes'].items():
    print(f"  Topic {topic}: {change:+d}")

# Create feedback visualizations
print("\nGenerating visualizations...")

# 1. Comprehensive feedback impact plot
fig1 = plot_feedback_impact(feedback_manager)
plt.figure(fig1.number)
plt.savefig("output/feedback_impact.png", dpi=150, bbox_inches="tight")
print(f"Saved feedback impact visualization to output/feedback_impact.png")

# 2. Topic distribution plot
fig2 = plot_topic_feedback_distribution(
    model,
    documents,
    original_topics,
    corrected_topics,
    show_wordclouds=True
)
plt.figure(fig2.number)
plt.savefig("output/topic_distribution_changes.png", dpi=150, bbox_inches="tight")
print(f"Saved topic distribution visualization to output/topic_distribution_changes.png")

# 3. Create interactive dashboard (commented out by default)
print("\nTo use the interactive dashboard, uncomment the dashboard code in this example.")
"""
# Create a copy of the model to represent "after" state
after_model = meno.TFIDFTopicModel(n_topics=5)
after_model.fit(documents)
# Update internal state to reflect corrected topics
after_model.doc_topics["topic"] = corrected_topics

# Create dashboard
app = create_feedback_comparison_dashboard(
    model,
    after_model,
    documents,
    changed_indices=[i for i, (a, b) in enumerate(zip(original_topics, corrected_topics)) if a != b]
)

# Run the dashboard
print("Running interactive dashboard on http://127.0.0.1:8050/")
app.run_server(debug=True)
"""

print("\nFeedback visualization example completed.")