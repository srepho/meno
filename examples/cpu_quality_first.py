#!/usr/bin/env python
# coding: utf-8

# # Quality-First CPU Topic Modeling with Meno
# 
# This script demonstrates how to prioritize quality over speed when running Meno on CPU-bound systems.
# It features robust API compatibility across different Meno versions and extends visualization options.
# 
# ## Usage Guide
#
# ### When to Use This Approach
# 
# This approach is ideal when:
# - Processing time is not a critical concern
# - You want the highest quality results possible
# - You don't have GPU acceleration available
# - You need superior topic separation and visualization
# - You need to ensure compatibility across Meno versions
# 
# ### Adapting to Your Tasks
# 
# To adapt this script for other CPU-bound topic modeling tasks:
# 
# 1. **Data Source**: Replace the `generate_quality_sample_data()` function with your data loading logic
#    - For CSV files: `df = pd.read_csv("your_data.csv")`
#    - For JSON: `df = pd.read_json("your_data.json")`
#    - For databases: Use appropriate connector and query
# 
# 2. **Configuration**: Modify the `QUALITY_CONFIG` dictionary to match your needs
#    - Adjust embedding model based on your domain needs
#    - Modify clustering parameters for your data characteristics
#    - Tune visualization settings for your specific analysis goals
# 
# 3. **Model Path**: Update `LOCAL_MODEL_PATH` to point to your cached model location
#    - Uncomment the auto-detection code to find locally cached models
#    - Or specify a custom path to your downloaded model
# 
# 4. **Task-Specific Visualizations**: Add custom visualization code in the visualization section
#    - For time series data: Add temporal analysis visualizations
#    - For geospatial data: Add map-based visualizations
#    - For hierarchical data: Enhance the topic hierarchy visualizations
# 
# ## Core Components Used
# 
# This script employs:
# - The full-featured `all-MiniLM-L6-v2` embedding model
# - UMAP dimensionality reduction (slower but better quality than PCA)
# - BERTopic with HDBSCAN clustering for optimal topic coherence
# - Detailed visualizations optimized for quality
# - Robust API compatibility across different Meno versions

# ## 1. Setup and Imports
# 
# First, let's import the necessary libraries and set up our environment.

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import meno if needed
parent_dir = str(Path().resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import meno components
from meno import MenoWorkflow, MenoTopicModeler
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel
from meno.visualization.bertopic_viz import create_bertopic_hierarchy


# ## 2. Configuration
# 
# Set up paths and configuration optimized for quality results.

# Set up paths and configuration
# Point to your downloaded model directory - update this path for your system
LOCAL_MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/24485cc25a8c8b310657ded9e17f6d18d1bdf0ae")

# You can also check if the model exists in the standard HuggingFace cache location
# Uncomment this code to automatically find the model in the cache
"""
try:
    cache_home = os.path.expanduser("~/.cache/huggingface/hub")
    model_files_dir = os.path.join(cache_home, "models--sentence-transformers--all-MiniLM-L6-v2")
    if os.path.exists(model_files_dir):
        # Find snapshots directory
        snapshots_dir = os.path.join(model_files_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                            if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                latest_snapshot = sorted(snapshot_dirs)[-1]
                LOCAL_MODEL_PATH = os.path.join(snapshots_dir, latest_snapshot)
                print(f"Found model in HuggingFace cache: {LOCAL_MODEL_PATH}")
            else:
                print("No snapshot directories found")
        else:
            print("Snapshots directory not found")
    else:
        print("Model directory not found in cache")
except Exception as e:
    print(f"Error finding local model: {e}")
"""

# Check if the specified path exists
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"WARNING: Model path {LOCAL_MODEL_PATH} does not exist!")
    print("Please update the LOCAL_MODEL_PATH to point to your downloaded model.")
else:
    print(f"Using model from: {LOCAL_MODEL_PATH}")

# Create output directory
OUTPUT_DIR = Path("./quality_output")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output will be saved to: {OUTPUT_DIR.absolute()}")

# Configure for quality-first CPU usage
QUALITY_CONFIG = {
    "preprocessing": {
        "normalization": {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "language": "en",
        },
    },
    "modeling": {
        "embeddings": {
            # Use best embedding model
            "model_name": "all-MiniLM-L6-v2",
            "local_model_path": LOCAL_MODEL_PATH,
            "local_files_only": True,
            
            # CPU settings (but not optimized for speed)
            "device": "cpu",
            "use_gpu": False,
            
            # Quality-focused settings
            "precision": "float32",  # Full precision for best quality
            "quantize": False,        # No quantization for best quality
            "batch_size": 16,         # Smaller batch size for better memory management
        },
        # High-quality HDBSCAN clustering settings
        "clustering": {
            "algorithm": "hdbscan",   # Explicitly set clustering algorithm
            "min_cluster_size": 5,
            "min_samples": 5,
            "prediction_data": True,
        },
        # For API compatibility
        "bertopic": {
            "min_topic_size": 5,
            "nr_topics": "auto"
        },
    },
    "visualization": {
        # High-quality UMAP settings
        "umap": {
            "n_neighbors": 15,  # Higher for more global structure
            "n_components": 3,   # 3D visualization
            "min_dist": 0.1,
            "metric": "cosine",
            "low_memory": False,  # Quality over memory efficiency
        },
        "plots": {
            "width": 1000,       # Larger plots for detail
            "height": 800,
            "template": "plotly_white",
        },
    },
}


# ## 3. Generate Sample Data
# 
# For this example, we'll generate synthetic data with subtle topic overlaps to demonstrate quality-focused modeling.

# Sample data generation function with more nuanced topics
def generate_quality_sample_data(n_samples=300):
    """Generate synthetic data for demonstration with subtle topic overlaps."""
    print(f"Generating {n_samples} sample documents with nuanced topics...")
    
    # Create topic templates with some overlapping terms
    topics = {
        "AI Technology": [
            "artificial intelligence neural networks deep learning algorithms training data",
            "machine learning models prediction classification regression computer vision",
            "natural language processing transformers bert gpt text generation tokens",
            "reinforcement learning agents environments rewards optimization policy"
        ],
        "Data Science": [
            "data analysis statistics regression visualization insights correlation",
            "big data processing pipelines hadoop spark streaming computation",
            "predictive modeling machine learning algorithms classification accuracy",
            "data science projects python pandas numpy visualization matplotlib"
        ],
        "Healthcare Analytics": [
            "medical data analysis patient outcomes treatment effectiveness metrics",
            "healthcare analytics prediction hospital readmission prevention care",
            "clinical decision support systems algorithms evidence patient data",
            "medical imaging analysis deep learning detection diagnosis pathology"
        ],
        "Financial Technology": [
            "fintech innovation banking technology digital payments blockchain",
            "algorithmic trading market prediction financial models risk analysis",
            "cryptocurrency blockchain transactions distributed ledger smart contracts",
            "financial data analysis machine learning fraud detection patterns"
        ],
        "Sustainable Energy": [
            "renewable energy solar wind hydroelectric power generation efficiency",
            "smart grid optimization data analysis consumption forecasting models",
            "energy storage technology batteries capacity efficiency innovation",
            "carbon emissions reduction monitoring data analysis climate impact"
        ]
    }
    
    # Create some cross-topic terms to make distinctions more subtle
    cross_topic_terms = {
        ("AI Technology", "Data Science"): 
            ["algorithms", "machine learning", "prediction", "models", "classification"],
        ("AI Technology", "Healthcare Analytics"): 
            ["medical imaging", "diagnosis", "prediction", "deep learning"],
        ("Data Science", "Financial Technology"): 
            ["data analysis", "prediction", "models", "algorithms"],
        ("AI Technology", "Sustainable Energy"): 
            ["optimization", "prediction", "models", "forecasting"],
        ("Data Science", "Healthcare Analytics"): 
            ["data analysis", "prediction", "patient data", "outcomes"]
    }
    
    # Generate documents from topics
    documents = []
    doc_ids = []
    doc_topics = []
    doc_subtopics = []
    
    topic_names = list(topics.keys())
    doc_id = 1
    
    for _ in range(n_samples):
        # Select a random primary topic
        primary_topic = np.random.choice(topic_names)
        doc_topics.append(primary_topic)
        
        # Select a random template
        template = np.random.choice(topics[primary_topic])
        words = template.split()
        
        # With some probability, add influence from another topic
        if np.random.random() < 0.3:  # 30% chance of topic overlap
            # Select a secondary topic that has cross-topic terms with the primary
            candidates = [t for t in topic_names if t != primary_topic and (primary_topic, t) in cross_topic_terms or (t, primary_topic) in cross_topic_terms]
            if candidates:
                secondary_topic = np.random.choice(candidates)
                doc_subtopics.append(secondary_topic)
                
                # Get cross-topic terms
                if (primary_topic, secondary_topic) in cross_topic_terms:
                    terms = cross_topic_terms[(primary_topic, secondary_topic)]
                else:
                    terms = cross_topic_terms[(secondary_topic, primary_topic)]
                
                # Add some cross-topic terms
                for term in np.random.choice(terms, size=min(3, len(terms)), replace=False):
                    words.append(term)
            else:
                doc_subtopics.append("None")
        else:
            doc_subtopics.append("None")
        
        # Create variations by adding noise and varying length
        num_words = len(words) + np.random.randint(-3, 10)
        if num_words < 5:
            num_words = 5
            
        # Select random words with replacement and shuffle for more realistic text
        selected_words = list(np.random.choice(words, size=num_words, replace=True))
        np.random.shuffle(selected_words)
        
        # Add some random transitional words for more natural text
        transitions = ["and", "also", "including", "with", "for", "about", "regarding", 
                      "related to", "concerning", "in terms of", "specifically"]
        for i in range(2, len(selected_words), 5):
            if i < len(selected_words):
                selected_words[i] = np.random.choice(transitions)
        
        document = " ".join(selected_words)
        documents.append(document)
        doc_ids.append(f"doc_{doc_id}")
        doc_id += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": documents,
        "id": doc_ids,
        "primary_topic": doc_topics,
        "secondary_topic": doc_subtopics
    })
    
    print(f"Generated {len(df)} documents across {len(topic_names)} primary topics")
    return df

# Generate the sample data
df = generate_quality_sample_data(n_samples=300)

# Display a few sample documents
print("\nSample documents:")
for topic in df["primary_topic"].unique():
    sample = df[df["primary_topic"] == topic].sample(1)
    secondary = sample["secondary_topic"].values[0]
    secondary_info = f" (with {secondary} influence)" if secondary != "None" else ""
    print(f"\n{topic}{secondary_info}: {sample['text'].values[0]}")

# Show topic distribution
print("\nPrimary topic distribution:")
print(df["primary_topic"].value_counts())

# Show secondary topic influence
print("\nSecondary topic influence:")
print(df["secondary_topic"].value_counts())

# Save the data for reference
df.to_csv(OUTPUT_DIR / "quality_sample_data.csv", index=False)
print(f"\nSample data saved to {OUTPUT_DIR / 'quality_sample_data.csv'}")


# ## 4. Initialize the Workflow
# 
# Now we'll set up the MenoWorkflow with quality-first settings.

# Initialize the workflow with quality-first settings
print("Initializing MenoWorkflow with quality-first settings...")
start_time = time.time()

workflow = MenoWorkflow(
    config_overrides=QUALITY_CONFIG,
    local_model_path=LOCAL_MODEL_PATH,
    local_files_only=True,
    offline_mode=True
)

# Load the data
workflow.load_data(data=df, text_column="text")
print(f"Loaded {len(df)} documents into the workflow")

# Measure initialization time
init_time = time.time() - start_time
print(f"Initialization completed in {init_time:.2f} seconds")


# ## 5. Preprocessing
# 
# Generate preprocessing reports and process the documents with thorough preprocessing.

# Start preprocessing timer
start_time = time.time()

print("Preprocessing documents with extensive cleaning...")
workflow.preprocess_documents(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemmatize=True,
    remove_numbers=True
)

# Get the preprocessed data
preprocessed_df = workflow.get_preprocessed_data()
print(f"Preprocessing completed for {len(preprocessed_df)} documents")

# Display sample of preprocessed text
print("\nSample of preprocessed text:")
sample_processed = preprocessed_df[["text", "processed_text"]].head(3)
print(sample_processed)

# Measure preprocessing time
preproc_time = time.time() - start_time
print(f"Preprocessing completed in {preproc_time:.2f} seconds")


# ## 6. High-Quality Topic Modeling with BERTopic
# 
# Run high-quality topic modeling using BERTopic with UMAP and HDBSCAN.

# Start topic modeling timer
start_time = time.time()

print("Discovering topics with high-quality settings (UMAP + HDBSCAN)...")
try:
    # First try with simple, newer API
    workflow.discover_topics(
        method="embedding_cluster",  # Use BERTopic for high-quality results
        modeling_approach="bertopic",
        num_topics=None  # Let HDBSCAN determine optimal topic count
    )
except Exception as e:
    print(f"First attempt failed: {e}, trying alternative API...")
    # Fallback to older API structure
    workflow.discover_topics(
        method="bertopic",
        nr_topics="auto",
        min_topic_size=5,
        auto_detect_topics=True
    )

# Try different API methods to get topic information
try:
    # First try newer API
    topics_df = workflow.get_topic_assignments()
    print("Using workflow.get_topic_assignments() API")
except Exception as e:
    # Fall back to direct document access
    print(f"Falling back to documents property: {e}")
    topics_df = workflow.modeler.documents  # Instead of get_topic_assignments()

# Try to get topic info from model
try:
    # Try to get topic info directly from model
    topic_info = workflow.modeler.get_topic_info()
    print("Using workflow.modeler.get_topic_info() API")
except Exception as e:
    # Manually create topic info dataframe
    print(f"Manually creating topic info: {e}")
    unique_topics = topics_df["topic"].unique()
    topic_counts = topics_df["topic"].value_counts().to_dict()
    
    # Create a simple topic info dataframe
    topic_info = pd.DataFrame([
        {"Topic": topic_id, 
         "Count": topic_counts.get(topic_id, 0),
         "Name": f"Topic {topic_id}" if topic_id != -1 else "Other"}
        for topic_id in unique_topics
    ])

print(f"\nDiscovered {len(topic_info) - 1 if -1 in unique_topics else len(topic_info)} meaningful topics automatically")

# Display topic distribution
print("\nTopic distribution:")
print(topic_info[['Topic', 'Count', 'Name']])  # Use 'Count' instead of 'Size'

# Display top words per topic
print("\nTop words per topic:")

# Create a function to extract top words for each topic from the documents
def get_top_words_for_topic(df, topic_id, top_n=10):
    # Get documents for this topic
    topic_docs = df[df['topic'] == topic_id]['processed_text']
    
    # Count word frequencies
    from collections import Counter
    word_counter = Counter()
    
    for doc in topic_docs:
        words = doc.split()
        word_counter.update(words)
    
    # Get top words
    return [word for word, count in word_counter.most_common(top_n)]

# Create function to format top words from model's get_topic_words method
def format_model_top_words(word_score_pairs):
    return [word for word, _ in word_score_pairs]

# Try to get top words using model method first, fall back to document extraction
for _, row in topic_info.iterrows():
    topic_id = row["Topic"]
    if topic_id != -1:  # Skip outlier topic
        try:
            # Try using model's get_topic_words method
            top_words = workflow.modeler.get_topic_words(topic_id, top_n=10)
            word_list = format_model_top_words(top_words)
            source = "model API"
        except Exception as e:
            # Fall back to extracting from documents
            word_list = get_top_words_for_topic(topics_df, topic_id)
            source = "document extraction"
        
        word_str = ", ".join(word_list)
        print(f"Topic {topic_id} ({row.get('Count', row.get('Size', 0))} docs): {word_str} (via {source})")

# Compare with ground truth
if "primary_topic" in df.columns:
    try:
        # Create merge key for joining
        if "id" not in topics_df.columns and "doc_id" in topics_df.columns:
            topics_df = topics_df.rename(columns={"doc_id": "id"})
            
        # Make sure we have a common key between the dataframes
        if "id" in topics_df.columns:
            # Get document assignment with original IDs
            doc_topics = topics_df.merge(df[["id", "primary_topic"]], on="id")
            
            # Show contingency table
            print("\nContingency table (Discovered vs. Actual):")
            contingency = pd.crosstab(doc_topics["topic"], doc_topics["primary_topic"])
            print(contingency)
            
            # Calculate adjusted mutual information
            from sklearn.metrics import adjusted_mutual_info_score
            ami = adjusted_mutual_info_score(
                doc_topics["topic"].apply(lambda x: str(x)), 
                doc_topics["primary_topic"]
            )
            print(f"\nAdjusted Mutual Information: {ami:.4f}")
        else:
            print("\nSkipping contingency table - no common ID column for joining.")
            # Try to compare in a different way
            actual_topic_counts = df["primary_topic"].value_counts()
            discovered_topic_counts = topics_df["topic"].value_counts()
            
            print("Actual topic distribution:")
            print(actual_topic_counts)
            print("\nDiscovered topic distribution:")
            print(discovered_topic_counts)
    except Exception as e:
        print(f"\nError creating contingency table: {e}")
        print("Continuing with the rest of the analysis...")

# Measure topic modeling time
topic_time = time.time() - start_time
print(f"\nTopic modeling completed in {topic_time:.2f} seconds")


# ## 7. Enhanced Topic Labeling
# 
# Create more descriptive topic labels based on the top keywords.

# Enhanced topic labeling function for better descriptions
def generate_enhanced_topic_label(topic_words, topic_id):
    """Generate a more descriptive topic label from keywords."""
    if topic_id == -1:
        return "Miscellaneous/Outliers"
    
    # Get key terms with weights
    keywords = [word for word, _ in topic_words[:5]]
    weights = [weight for _, weight in topic_words[:5]]
    
    # Check for specific domain indicators
    domains = {
        "ai": ["ai", "artificial", "intelligence", "machine", "learning", "neural", "deep"],
        "healthcare": ["medical", "health", "patient", "clinical", "hospital", "doctor"],
        "finance": ["financial", "finance", "banking", "investment", "market", "trading"],
        "energy": ["energy", "power", "renewable", "sustainable", "carbon", "grid"],
        "data": ["data", "analysis", "analytics", "processing", "visualization"]
    }
    
    # Check if keywords match any domains
    domain_matches = {}
    for domain, terms in domains.items():
        matches = sum(1 for kw in keywords if kw in terms)
        if matches > 0:
            domain_matches[domain] = matches
    
    # If we have domain matches, use the best one as prefix
    if domain_matches:
        best_domain = max(domain_matches.items(), key=lambda x: x[1])[0]
        domain_prefix = {
            "ai": "AI & ",
            "healthcare": "Healthcare & ",
            "finance": "Financial & ",
            "energy": "Energy & ",
            "data": "Data Science & "
        }[best_domain]
    else:
        domain_prefix = ""
    
    # Combine top keywords with weights for emphasis
    primary_keywords = [keywords[0], keywords[1]] if len(keywords) > 1 else [keywords[0]]
    secondary_keywords = keywords[2:4] if len(keywords) > 3 else keywords[2:]
    
    # Format the label based on available keywords
    if secondary_keywords:
        label = f"{domain_prefix}{' & '.join(primary_keywords).title()} ({', '.join(secondary_keywords)})"
    else:
        label = f"{domain_prefix}{' & '.join(primary_keywords).title()}"
    
    return label

# Generate enhanced topic labels
print("Generating enhanced topic labels...")
topic_labels = {}

for topic_id in topic_info["Topic"].unique():
    # Try getting top words using model API first, then fallback to document extraction
    if topic_id != -1:
        try:
            # Try model method first
            top_words = workflow.modeler.get_topic_words(topic_id, top_n=10)
            source = "model API"
        except Exception as e:
            # Fall back to document extraction
            top_words_list = get_top_words_for_topic(topics_df, topic_id)
            # Convert to format expected by generate_enhanced_topic_label
            top_words = [(word, 1.0) for word in top_words_list]
            source = "document extraction"
        
        print(f"Got top words for Topic {topic_id} via {source}")
    else:
        top_words = []  # Empty for outlier topic
    
    # Generate enhanced label
    label = generate_enhanced_topic_label(top_words, topic_id)
    topic_labels[topic_id] = label

# Print enhanced labels
print("\nEnhanced topic labels:")
for topic_id, label in topic_labels.items():
    if topic_id != -1:  # Skip outlier topic in display
        topic_size = topic_info[topic_info["Topic"] == topic_id]["Count"].values[0]
        print(f"Topic {topic_id} ({topic_size} docs): {label}")

# Save topic information with enhanced labels
topic_info["Enhanced_Label"] = topic_info["Topic"].map(topic_labels)
topic_info.to_csv(OUTPUT_DIR / "topic_summary_enhanced.csv", index=False)
print(f"\nEnhanced topic summary saved to {OUTPUT_DIR / 'topic_summary_enhanced.csv'}")

# Save document-topic assignments with enhanced labels
topics_df["Topic_Label"] = topics_df["topic"].map(topic_labels)
topics_df.to_csv(OUTPUT_DIR / "document_topics_enhanced.csv", index=False)
print(f"Document-topic assignments saved to {OUTPUT_DIR / 'document_topics_enhanced.csv'}")


# ## 8. Generate Comprehensive HTML Report
# 
# Create a high-quality HTML report with all the topic modeling results.

# Start report generation timer
start_time = time.time()

print("Generating comprehensive report with enhanced visualizations...")
try:
    # First try with topic_labels parameter
    report_path = workflow.generate_comprehensive_report(
        output_path=OUTPUT_DIR / "high_quality_topic_report.html",
        open_browser=False,
        title="High-Quality CPU Topic Analysis Report",
        include_interactive=True,
        topic_labels=topic_labels
    )
    print("Generated report with custom topic labels")
except Exception as e:
    print(f"First report attempt failed: {e}, trying without topic_labels parameter")
    try:
        # Try again without topic_labels parameter
        report_path = workflow.generate_comprehensive_report(
            output_path=OUTPUT_DIR / "high_quality_topic_report.html",
            open_browser=False,
            title="High-Quality CPU Topic Analysis Report",
            include_interactive=True
        )
        print("Generated report without custom topic labels")
    except Exception as e2:
        print(f"Error generating comprehensive report: {e2}")
        report_path = None

print(f"Report generated at {report_path}")

# Measure report generation time
report_time = time.time() - start_time
print(f"Report generation completed in {report_time:.2f} seconds")


# ## 9. High-Quality BERTopic Visualizations
# 
# Create detailed UMAP-based visualizations that prioritize quality.

# Start visualization timer
start_time = time.time()

print("Creating high-quality UMAP visualizations (this may take some time)...")

# Try multiple visualization methods with graceful fallbacks
print("\nGenerating visualizations with multiple approaches...")

# 1. Try UMAP embedding visualization with rich parameters first
try:
    print("\nGenerating 3D UMAP visualization with enhanced parameters...")
    embed_fig = workflow.modeler.visualize_embeddings(
        return_figure=True,
        plot_3d=True,  # 3D visualization
        width=1000,
        height=800,
        include_topic_labels=True,
        hover_data=["topic", "Topic_Label"],
        topic_label_dict=topic_labels
    )
    embed_fig.write_html(OUTPUT_DIR / "3d_topic_embeddings.html")
    print(f"Enhanced 3D UMAP visualization saved to {OUTPUT_DIR / '3d_topic_embeddings.html'}")
except Exception as e:
    print(f"Enhanced visualization failed: {e}, trying simpler visualization...")
    
    # 2. Fallback to simpler embedding visualization
    try:
        print("\nGenerating basic UMAP visualization...")
        embed_fig = workflow.modeler.visualize_embeddings(
            return_figure=True
            # No additional parameters
        )
        embed_fig.write_html(OUTPUT_DIR / "topic_embeddings.html")
        print(f"Basic UMAP visualization saved to {OUTPUT_DIR / 'topic_embeddings.html'}")
    except Exception as e2:
        print(f"Could not create UMAP visualization: {e2}")

# 3. Try to access BERTopic model for advanced visualizations
try:
    print("\nAttempting to access BERTopic model for advanced visualizations...")
    model = workflow.modeler.topic_model
    
    # Try topic similarity network
    try:
        print("Generating topic similarity network...")
        network_fig = model.visualize_topics(
            topics="all",
            top_n_topics=None
        )
        network_fig.write_html(OUTPUT_DIR / "topic_similarity_network.html")
        print(f"Topic similarity network saved to {OUTPUT_DIR / 'topic_similarity_network.html'}")
    except Exception as e:
        print(f"Could not create topic similarity network: {e}")
        
except Exception as e:
    print(f"Could not access BERTopic model directly: {e}")
    print("Skipping BERTopic-specific visualizations which require direct model access.")

# Create extended visualizations for different types of analysis
try:
    print("\nGenerating extended visualization suite...")
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    # =======================================================================
    # 1. WORD DISTRIBUTION VISUALIZATIONS
    # =======================================================================
    print("\n1. Creating word distribution visualizations...")
    
    # For the top 5 topics, create bar charts of top words
    for topic_id in [t for t in topic_info["Topic"].unique() if t != -1][:5]:
        # Get top words with counts
        topic_docs = workflow.modeler.documents[workflow.modeler.documents['topic'] == topic_id]['processed_text']
        word_counter = {}
        for doc in topic_docs:
            words = doc.split()
            for word in words:
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1
        
        # Sort and get top words
        sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)[:15]
        words = [word for word, _ in sorted_words]
        counts = [count for _, count in sorted_words]
        
        # Create enhanced bar chart with custom styling
        fig = go.Figure()
        
        # Add main bar chart
        fig.add_trace(go.Bar(
            x=words,
            y=counts,
            text=counts,
            textposition='auto',
            marker_color='rgba(58, 71, 180, 0.7)',
            marker_line_color='rgba(8, 48, 107, 1.0)',
            marker_line_width=1
        ))
        
        fig.update_layout(
            title={
                'text': f"Top Words for {topic_labels.get(topic_id, f'Topic {topic_id}')}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Words",
            yaxis_title="Frequency",
            width=1000,
            height=600,
            template="plotly_white",
            # Add a subtle grid for better readability
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='lightgray',
                tickangle=-45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.8)',
                showline=True,
                linecolor='lightgray'
            )
        )
        
        fig.write_html(OUTPUT_DIR / f"topic_{topic_id}_word_distribution.html")
        print(f"Enhanced word distribution for Topic {topic_id} saved")
    
    # =======================================================================
    # 2. TOPIC PROPORTION VISUALIZATION 
    # =======================================================================
    print("\n2. Creating topic proportion visualization...")
    
    # Create a pie chart showing distribution of topics
    topic_counts = topics_df["topic"].value_counts().sort_index()
    topic_names = [topic_labels.get(topic_id, f"Topic {topic_id}") for topic_id in topic_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=topic_names,
        values=topic_counts.values,
        hole=.4,
        textinfo='percent+label',
    )])
    
    fig.update_layout(
        title="Distribution of Topics",
        width=900,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.write_html(OUTPUT_DIR / "topic_proportion_distribution.html")
    print("Topic proportion visualization saved")
    
    # =======================================================================
    # 3. TOPIC COMPARISON VISUALIZATION
    # =======================================================================
    print("\n3. Creating topic comparison visualization...")
    
    # Compare top words across topics in a single visualization
    # Select up to 4 topics for comparison
    topics_to_compare = [t for t in topic_info["Topic"].unique() if t != -1][:min(4, len(topic_info))]
    
    # Create subplots
    fig = make_subplots(rows=len(topics_to_compare), cols=1, 
                       subplot_titles=[topic_labels.get(t, f"Topic {t}") for t in topics_to_compare],
                       vertical_spacing=0.1)
    
    # Add bars for each topic
    for i, topic_id in enumerate(topics_to_compare):
        # Get top words
        try:
            words = get_top_words_for_topic(topics_df, topic_id, top_n=10)
            if not words:
                continue
                
            # Count occurrences
            topic_docs = topics_df[topics_df['topic'] == topic_id]['processed_text']
            word_counts = {}
            for word in words:
                word_counts[word] = sum(1 for doc in topic_docs if word in doc.split())
            
            # Sort by count
            sorted_items = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            words = [item[0] for item in sorted_items]
            counts = [item[1] for item in sorted_items]
            
            # Use different colors for each topic
            colors = ['rgba(31, 119, 180, 0.7)', 'rgba(255, 127, 14, 0.7)', 
                     'rgba(44, 160, 44, 0.7)', 'rgba(214, 39, 40, 0.7)']
            
            fig.add_trace(
                go.Bar(
                    x=words,
                    y=counts,
                    name=f"Topic {topic_id}",
                    marker_color=colors[i % len(colors)]
                ),
                row=i+1, col=1
            )
        except Exception as e:
            print(f"Could not create comparison for topic {topic_id}: {e}")
    
    fig.update_layout(
        height=300 * len(topics_to_compare),
        width=900,
        title_text="Topic Word Comparison",
        showlegend=False
    )
    
    fig.write_html(OUTPUT_DIR / "topic_comparison.html")
    print("Topic comparison visualization saved")
    
    # =======================================================================
    # 4. INTERACTIVE DOCUMENT-TOPIC EXPLORER
    # =======================================================================
    print("\n4. Creating interactive document-topic explorer...")
    
    # Create a scatter plot of documents colored by topic
    try:
        # Add length of document as a feature
        topics_df['doc_length'] = topics_df['processed_text'].apply(lambda x: len(x.split()))
        
        # Select a sample of documents (to avoid cluttering)
        sample_size = min(300, len(topics_df))
        df_sample = topics_df.sample(sample_size)
        
        # Create a scatter plot
        fig = px.scatter(
            df_sample, 
            x='doc_length', 
            y='topic', 
            color='topic', 
            hover_data=['text'],
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.G10,
            labels={
                'doc_length': 'Document Length (words)',
                'topic': 'Topic ID',
                'text': 'Document Text'
            }
        )
        
        # Update layout
        fig.update_layout(
            title="Document-Topic Explorer",
            xaxis_title="Document Length (words)",
            yaxis_title="Topic ID",
            width=1000,
            height=700,
            template="plotly_white"
        )
        
        # Add custom hover template to show excerpt of text
        fig.update_traces(
            hovertemplate='<b>Topic:</b> %{y}<br><b>Doc Length:</b> %{x}<br><b>Text:</b> %{customdata[0]:.60}...'
        )
        
        fig.write_html(OUTPUT_DIR / "document_topic_explorer.html")
        print("Document-topic explorer saved")
    except Exception as e:
        print(f"Could not create document-topic explorer: {e}")
        
except Exception as e:
    print(f"Could not create extended visualizations: {e}")

# Measure visualization time
viz_time = time.time() - start_time
print(f"\nHigh-quality visualizations completed in {viz_time:.2f} seconds")


# ## 10. Advanced Topic Analysis
# 
# Perform deeper analysis of the topic structure and document assignments.

# Start advanced analysis timer
start_time = time.time()

print("Performing advanced topic analysis...")

# Extract and save representative documents per topic - without using model.get_representative_docs
representative_docs = []

# Find representative documents based on number of topic keywords they contain
for topic_id in [t for t in topic_info["Topic"].unique() if t != -1]:  # Skip outlier topic
    try:
        # Get documents for this topic
        topic_docs_df = workflow.modeler.documents[workflow.modeler.documents['topic'] == topic_id]
        
        # Get top words for this topic
        top_words = get_top_words_for_topic(workflow.modeler.documents, topic_id, top_n=20)
        
        # Score documents by how many top words they contain
        doc_scores = []
        for idx, row in topic_docs_df.iterrows():
            doc_text = row['text']
            processed_text = row['processed_text']
            
            # Count matches of top words
            word_matches = sum(1 for word in top_words if word in processed_text.split())
            
            doc_scores.append((idx, doc_text, word_matches))
        
        # Sort by score and get top 3
        top_docs = sorted(doc_scores, key=lambda x: x[2], reverse=True)[:3]
        
        # Add to representative docs
        for _, doc_text, _ in top_docs:
            representative_docs.append({
                "topic_id": topic_id,
                "topic_label": topic_labels.get(topic_id, f"Topic {topic_id}"),
                "document": doc_text
            })
    except Exception as e:
        print(f"Could not get representative docs for Topic {topic_id}: {e}")
        continue

# Create DataFrame of representative documents
if representative_docs:
    rep_docs_df = pd.DataFrame(representative_docs)
    rep_docs_df.to_csv(OUTPUT_DIR / "representative_documents.csv", index=False)
    print(f"\nSaved {len(representative_docs)} representative documents to {OUTPUT_DIR / 'representative_documents.csv'}")
    
    # Display a sample
    print("\nSample representative documents:")
    print(rep_docs_df.groupby("topic_label").head(1)[["topic_label", "document"]])

# Calculate topic similarity matrix using multiple approaches
try:
    print("\nCalculating topic similarity matrix...")
    topic_ids = [t for t in topic_info["Topic"].unique() if t != -1]
    similarity_matrix = np.zeros((len(topic_ids), len(topic_ids)))
    
    # Try to access model's topic vectors first
    try:
        print("Attempting to use model's topic vectors...")
        model = workflow.modeler.topic_model
        
        for i, topic1 in enumerate(topic_ids):
            for j, topic2 in enumerate(topic_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Get topic vectors
                    vector1 = model.topic_vectors.get(topic1, None)
                    vector2 = model.topic_vectors.get(topic2, None)
                    
                    if vector1 is not None and vector2 is not None:
                        # Calculate cosine similarity
                        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                        similarity_matrix[i, j] = similarity
                    else:
                        similarity_matrix[i, j] = 0.0
        
        print("Successfully calculated similarity using model's topic vectors")
    except Exception as e:
        print(f"Couldn't use model vectors: {e}, falling back to word overlap approach...")
        
        # Fallback to word overlap approach
        # Get word sets for each topic
        topic_word_sets = {}
        for topic_id in topic_ids:
            try:
                # Try to get topic words from model
                words = workflow.modeler.get_topic_words(topic_id, top_n=50)
                top_words = [word for word, _ in words]
            except Exception:
                # Fall back to document extraction
                top_words = get_top_words_for_topic(topics_df, topic_id, top_n=50)
                
            topic_word_sets[topic_id] = set(top_words)
        
        # Calculate Jaccard similarity between topic word sets
        for i, topic1 in enumerate(topic_ids):
            for j, topic2 in enumerate(topic_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    word_set1 = topic_word_sets[topic1]
                    word_set2 = topic_word_sets[topic2]
                    
                    # Jaccard similarity: intersection over union
                    intersection = len(word_set1.intersection(word_set2))
                    union = len(word_set1.union(word_set2))
                    
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0
                        
        print("Successfully calculated similarity using word overlap approach")
    
    # Create similarity DataFrame with enhanced labels
    topic_labels_list = [topic_labels.get(t, f"Topic {t}") for t in topic_ids]
    similarity_df = pd.DataFrame(similarity_matrix, index=topic_labels_list, columns=topic_labels_list)
    
    # Save to CSV
    similarity_df.to_csv(OUTPUT_DIR / "topic_similarity_matrix.csv")
    print(f"Topic similarity matrix saved to {OUTPUT_DIR / 'topic_similarity_matrix.csv'}")
    
    # Create heatmap visualization of topic similarity
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=topic_labels_list,
        y=topic_labels_list,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Topic Similarity Matrix",
        width=1000,
        height=800,
        xaxis=dict(title="Topic"),
        yaxis=dict(title="Topic")
    )
    
    fig.write_html(OUTPUT_DIR / "topic_similarity_heatmap.html")
    print(f"Topic similarity heatmap saved to {OUTPUT_DIR / 'topic_similarity_heatmap.html'}")
    
except Exception as e:
    print(f"Could not calculate topic similarity matrix: {e}")

# Measure advanced analysis time
analysis_time = time.time() - start_time
print(f"\nAdvanced analysis completed in {analysis_time:.2f} seconds")


# ## 11. Performance Summary
# 
# Summarize the performance metrics of our quality-focused workflow.

# Create performance summary
print("Performance Summary")
print("===================\n")
print(f"Dataset size: {len(df)} documents")
print(f"Topics discovered: {len(topic_info) - 1} (excluding outliers)")
print("\nProcessing times:")
print(f"- Initialization: {init_time:.2f} seconds")
print(f"- Preprocessing: {preproc_time:.2f} seconds")
print(f"- Topic modeling: {topic_time:.2f} seconds")
print(f"- Report generation: {report_time:.2f} seconds")
print(f"- Visualizations: {viz_time:.2f} seconds")
print(f"- Advanced analysis: {analysis_time:.2f} seconds")
print(f"- Total processing time: {init_time + preproc_time + topic_time + report_time + viz_time + analysis_time:.2f} seconds")

if "primary_topic" in df.columns and 'ami' in locals():
    print(f"\nAdjusted Mutual Information: {ami:.4f}")


# ## 12. Summary and Conclusion
# 
# This script demonstrated a quality-first approach to CPU-bound topic modeling using Meno with robust API compatibility. We prioritized result quality over processing speed by using:
# 
# 1. **Full-featured embedding models**: Using `all-MiniLM-L6-v2` without quantization for best embedding quality
# 2. **UMAP dimensionality reduction**: Slower but produces superior topic separation vs. PCA
# 3. **High-quality BERTopic**: Using HDBSCAN clustering for optimal topic coherence
# 4. **Enhanced visualizations**: Creating detailed, information-rich visualizations
# 5. **Advanced topic analysis**: Performing deeper analysis of topic structure and relationships
# 
# ### Key Benefits of this Approach
# 
# - **Superior topic separation**: Better distinguishes between related topics
# - **Higher topic coherence**: Topics are more internally consistent
# - **More detailed visualizations**: Richer visual representations of topic relationships
# - **Enhanced topic labels**: More meaningful descriptions of topic content
# - **Works entirely on CPU**: No GPU required, just more processing time
# 
# ### API Compatibility Features
# 
# This script includes several compatibility mechanisms to work with both newer and older Meno APIs:
# 
# 1. **Multiple API support**: Graceful fallbacks between different API versions
# 2. **Error handling**: Robust try/except blocks to handle API differences
# 3. **Flexible configuration**: Compatible settings for different Meno versions
# 4. **Alternative implementations**: Custom methods when direct API access isn't available
# 5. **Progressive enhancement**: Attempts richer features first, falls back to simpler ones
# 
# ### Visualization Extensions
# 
# The script provides several types of visualizations for different analytical needs:
# 
# 1. **Enhanced Word Distribution**: Improved bar charts with better styling for top words per topic
# 2. **Topic Proportion Distribution**: Interactive pie charts showing relative sizes of topics
# 3. **Multi-Topic Comparison**: Side-by-side comparison of key words across topics
# 4. **Document-Topic Explorer**: Interactive scatter plot to explore documents by topic and length
# 
# ### Adapting to Other Domain-Specific Topic Modeling Tasks
# 
# To adapt this script for other domains:
# 
# - **Medical Text Analysis**: Adjust preprocessing to handle medical terminology; consider using domain-specific embeddings
# - **Legal Document Processing**: Increase `min_cluster_size` for longer documents; add specialized visualizations for legal citations
# - **Customer Feedback Analysis**: Add sentiment analysis components; create time-based topic evolution visualizations
# - **Academic Literature Review**: Enhance citation tracking; add author and journal metadata to topic exploration
# - **Financial Document Analysis**: Add named entity recognition for financial terms; create entity-topic relationship visualizations
# 
# This approach is ideal when you prioritize result quality over processing speed, especially for more nuanced datasets where topics have subtle differences or overlap, and need to ensure compatibility across different Meno versions.