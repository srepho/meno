#!/usr/bin/env python
"""
Generate enhanced sample reports demonstrating all new visualization features.
This script creates synthetic data and generates comprehensive HTML reports.
"""

import numpy as np
import pandas as pd
import os
import random
from pathlib import Path
from datetime import datetime, timedelta

# Import meno components
from meno.meno import MenoTopicModeler
from meno.reporting.html_generator import generate_html_report

# Set output directory
OUTPUT_DIR = Path("examples/sample_reports/enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create synthetic data
def generate_synthetic_data(n_docs=200, n_topics=5, coherence=0.8, random_seed=42):
    """Generate synthetic documents with topics for demo purposes."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create topic words
    topic_words = {
        f"Topic_{i+1}": {
            f"word_{i}_{j}": np.random.gamma(shape=2.0, scale=2.0) 
            for j in range(1, 51)
        } for i in range(n_topics)
    }
    
    # Add some common words to make topics more realistic
    common_words = {
        "Topic_1": {"insurance": 9.5, "claim": 8.7, "policy": 7.9, "damage": 7.1, "vehicle": 6.5, 
                   "accident": 6.2, "car": 5.8, "coverage": 5.5, "report": 5.1, "collision": 4.8},
        "Topic_2": {"water": 9.6, "damage": 8.9, "flood": 8.2, "pipe": 7.5, "leak": 7.0, 
                   "home": 6.5, "basement": 6.0, "repair": 5.7, "mold": 5.4, "property": 5.1},
        "Topic_3": {"injury": 9.6, "medical": 8.8, "pain": 8.1, "treatment": 7.4, "hospital": 7.0, 
                   "doctor": 6.5, "accident": 6.1, "therapy": 5.8, "recovery": 5.3, "diagnosis": 5.0},
        "Topic_4": {"fire": 9.5, "home": 8.8, "damage": 8.3, "property": 7.7, "smoke": 7.2, 
                   "kitchen": 6.8, "electrical": 6.3, "alarm": 5.9, "flames": 5.5, "emergency": 5.2},
        "Topic_5": {"theft": 9.7, "stolen": 9.0, "property": 8.3, "break-in": 7.8, "burglary": 7.2, 
                   "items": 6.7, "home": 6.3, "security": 5.9, "police": 5.5, "report": 5.2}
    }
    
    # Add common words to topic_words
    for topic in common_words:
        if topic in topic_words:
            topic_words[topic].update(common_words[topic])
    
    # Sample documents for each topic
    docs_per_topic = n_docs // n_topics
    remaining_docs = n_docs % n_topics
    
    documents = []
    topics = []
    
    # Sample example texts for each topic
    example_texts = {
        "Topic_1": [
            "Customer's vehicle was damaged in a collision at an intersection. Front bumper and headlight need replacement.",
            "Policyholder reported a car accident where their vehicle was struck from behind, causing rear damage.",
            "Claimant's car sustained hail damage during the storm last week. Multiple dents on hood and roof.",
            "Vehicle slid on ice and hit a guardrail. Damage to driver's side door and panel. No injuries reported.",
            "Customer's parked car was hit by another vehicle in a parking lot. Passenger side damage visible."
        ],
        "Topic_2": [
            "Water damage in basement due to broken pipe. Carpet and drywall affected. Mold starting to form.",
            "Flooding from heavy rain entered first floor of home. Hardwood floors and furniture damaged.",
            "Kitchen sink pipe burst while homeowner was away. Significant water damage to cabinets and floor.",
            "Water heater leak caused damage to utility room and adjacent hallway carpet.",
            "Dishwasher malfunction resulted in water damage to kitchen floor and baseboards."
        ],
        "Topic_3": [
            "Claimant slipped on wet floor at grocery store resulting in back injury. Seeking medical treatment.",
            "Customer fell down stairs and injured wrist. X-ray confirmed fracture. Treatment and therapy needed.",
            "Workplace injury when lifting heavy equipment. Lower back pain requiring ongoing physical therapy.",
            "Pedestrian hit by bicycle sustaining minor injuries including bruising and sprained ankle.",
            "Repetitive strain injury from computer work affecting wrists and hands. Diagnosis of carpal tunnel."
        ],
        "Topic_4": [
            "Kitchen fire damaged cabinets and appliances. Smoke damage throughout first floor of home.",
            "Electrical fire in living room wall. Fire department responded. Smoke and water damage to room.",
            "Small grease fire on stove damaged range hood and caused smoke damage to kitchen ceiling.",
            "Lightning strike caused attic fire. Roof damage and smoke damage to upstairs bedrooms.",
            "Fire damage to detached garage from malfunctioning space heater. Vehicle inside also damaged."
        ],
        "Topic_5": [
            "Home break-in through back window. Electronics and jewelry stolen. Police report filed.",
            "Theft of personal items from vehicle parked in driveway. Window broken to gain access.",
            "Bicycle stolen from garage. Door was unlocked. Security camera captured suspect image.",
            "Package theft from front porch. Delivery confirmed but items missing upon arrival home.",
            "Office break-in with theft of computers and equipment. Forced entry through rear door."
        ]
    }
    
    # Generate document data
    for i in range(n_topics):
        topic_name = f"Topic_{i+1}"
        n_docs_this_topic = docs_per_topic + (1 if i < remaining_docs else 0)
        
        examples = example_texts.get(topic_name, [f"Example document for {topic_name}"])
        
        for j in range(n_docs_this_topic):
            # Choose a base example document
            base_doc = random.choice(examples)
            
            # Create a document ID
            doc_id = f"doc_{i}_{j}"
            
            # Timestamp for time-based analysis
            date = datetime.now() - timedelta(days=random.randint(0, 365))
            
            documents.append({
                "doc_id": doc_id,
                "text": base_doc,
                "processed_text": " ".join([w for w in base_doc.lower().split() if len(w) > 3]),
                "topic": topic_name,
                "topic_probability": np.random.beta(a=10*coherence, b=10*(1-coherence)),
                "coherence": coherence + np.random.normal(0, 0.05),
                "date": date
            })
            topics.append(topic_name)

    # Create DataFrame
    df = pd.DataFrame(documents)
    
    # Create topic assignments dataframe with similarity columns
    topic_assignments = df[["doc_id", "topic", "topic_probability", "coherence"]].copy()
    
    # Add similarity columns
    for t in range(1, n_topics+1):
        topic_name = f"Topic_{t}"
        topic_assignments[f"{topic_name}_similarity"] = np.where(
            topic_assignments["topic"] == topic_name,
            topic_assignments["topic_probability"],
            np.random.beta(a=1, b=10)
        )
    
    # Create "top_words" column with comma-separated words
    def get_top_words(topic):
        if topic in topic_words:
            words = sorted(topic_words[topic].items(), key=lambda x: x[1], reverse=True)
            return ", ".join(word for word, _ in words[:10])
        return ""
    
    topic_assignments["top_words"] = topic_assignments["topic"].apply(get_top_words)
    
    # Create 2D embeddings for visualization
    embeddings = np.zeros((len(df), 2))
    
    # Generate cluster-like embeddings
    centers = {
        f"Topic_{i+1}": np.array([
            np.cos(2 * np.pi * i / n_topics) * 10,
            np.sin(2 * np.pi * i / n_topics) * 10
        ]) for i in range(n_topics)
    }
    
    for i, topic in enumerate(df["topic"]):
        # Base position from the topic center
        center = centers[topic]
        # Add noise
        noise = np.random.normal(0, 2, 2)
        embeddings[i] = center + noise
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((n_topics, n_topics))
    
    for i in range(n_topics):
        for j in range(n_topics):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Higher similarity for adjacent topics
                dist = min(abs(i - j), n_topics - abs(i - j))
                similarity_matrix[i, j] = max(0.1, 1.0 - (dist / n_topics) - 0.1 * np.random.random())
    
    return df, topic_assignments, embeddings, topic_words, similarity_matrix

# Generate data
docs_df, topic_assignments_df, embeddings, topic_words, similarity_matrix = generate_synthetic_data(
    n_docs=250, 
    n_topics=5
)

# Save sample of data for demonstration
docs_sample = docs_df.head(10)
docs_sample[["doc_id", "text", "topic", "topic_probability"]].to_csv(
    OUTPUT_DIR / "sample_documents.csv", 
    index=False
)

print(f"Generated {len(docs_df)} synthetic documents across {len(topic_words)} topics")

# Generate comprehensive report with all features
report_path = generate_html_report(
    documents=docs_df,
    topic_assignments=topic_assignments_df,
    umap_projection=embeddings,
    output_path=OUTPUT_DIR / "comprehensive_report.html",
    config={
        "title": "Enhanced Topic Modeling Report",
        "include_interactive": True,
        "max_examples_per_topic": 5,
        "include_raw_data": True,
    },
    similarity_matrix=similarity_matrix,
    topic_words=topic_words,
)

print(f"Generated comprehensive report at {report_path}")

# Generate report with similarity matrix but no word clouds
report_path = generate_html_report(
    documents=docs_df,
    topic_assignments=topic_assignments_df,
    umap_projection=embeddings,
    output_path=OUTPUT_DIR / "similarity_focused_report.html",
    config={
        "title": "Topic Similarity Analysis",
        "include_interactive": True,
        "max_examples_per_topic": 3,
        "include_raw_data": False,
    },
    similarity_matrix=similarity_matrix,
    topic_words=None,
)

print(f"Generated similarity-focused report at {report_path}")

# Generate report with word clouds but no similarity matrix
report_path = generate_html_report(
    documents=docs_df,
    topic_assignments=topic_assignments_df,
    umap_projection=embeddings,
    output_path=OUTPUT_DIR / "wordcloud_focused_report.html",
    config={
        "title": "Topic Word Distribution Analysis",
        "include_interactive": True,
        "max_examples_per_topic": 3,
        "include_raw_data": False,
    },
    similarity_matrix=None,
    topic_words=topic_words,
)

print(f"Generated word cloud-focused report at {report_path}")

# Generate report using the MenoTopicModeler class
modeler = MenoTopicModeler()

# Set the documents and topic assignments directly 
modeler.documents = docs_df
modeler.topic_assignments = topic_assignments_df
modeler.umap_projection = embeddings

# Generate report with all enhancements
report_path = modeler.generate_report(
    output_path=OUTPUT_DIR / "modeler_enhanced_report.html",
    title="MenoTopicModeler Enhanced Report",
    include_interactive=True,
    include_raw_data=True,
    max_examples_per_topic=4,
    similarity_matrix=similarity_matrix,
    topic_words=topic_words
)

print(f"Generated report using MenoTopicModeler at {report_path}")

print("\nAll sample reports have been generated in the examples/sample_reports/enhanced directory")