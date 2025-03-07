#!/usr/bin/env python
"""
Consistent API Demo

This script demonstrates the consistent API design in Meno,
showing how parameters and return values are standardized
across different components, making it easier to use.

Key features demonstrated:
1. Consistent parameter naming across model types
2. Uniform return types for similar methods
3. Standardized visualization parameters
4. Type hints for better IDE integration
5. Predictable behavior across models
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("meno_api_demo")

# Add the parent directory to path for direct execution
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Create output directory
output_dir = Path("output/consistent_api_demo")
output_dir.mkdir(exist_ok=True, parents=True)

# Function to demonstrate and document API consistency
def document_consistent_api(
    method_name: str,
    similar_methods: List[str],
    common_parameters: List[str],
    common_return_type: str,
    example_code: str
) -> None:
    """Document consistency between related API methods."""
    
    print(f"\n📘 {method_name} API Consistency")
    print("=" * (len(method_name) + 18))
    print(f"Related methods: {', '.join(similar_methods)}")
    print(f"Consistent parameters across methods: {', '.join(common_parameters)}")
    print(f"Common return type: {common_return_type}")
    
    print("\nExample usage:")
    print("```python")
    print(example_code)
    print("```")

# Import all necessary Meno components
from meno.modeling.embeddings import EmbeddingModel
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.top2vec_model import Top2VecModel
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler, create_topic_modeler
from meno.visualization.static_plots import create_topic_distribution_plot
from meno.visualization.interactive_plots import create_interactive_topic_distribution
from meno.visualization.umap_viz import create_embedding_visualization
from meno.workflow import MenoWorkflow

print("🔍 Meno Consistent API Demonstration")
print("===================================")
print("This script showcases the consistent API design patterns used in Meno.")

# Create sample data for demonstration
print("\n📊 Generating sample data...")
sample_docs = [
    "Machine learning models require careful tuning and evaluation.",
    "Deep neural networks have revolutionized natural language processing.",
    "Data preprocessing is essential for effective model training.",
    "Feature engineering improves model performance significantly.",
    "Model validation prevents overfitting to training data.",
    "Hyperparameter optimization helps find the best model configuration.",
    "Transfer learning leverages pre-trained models for new tasks.",
    "Model interpretability is crucial for responsible AI deployment.",
    "Ensemble methods combine multiple models for better predictions.",
    "Cross-validation provides robust performance estimates."
]

sample_df = pd.DataFrame({
    "text": sample_docs,
    "id": list(range(len(sample_docs))),
    "category": ["ML"] * 3 + ["DL"] * 2 + ["ML"] * 3 + ["DL"] * 2
})

# 1. Document consistent embedding model interface
document_consistent_api(
    method_name="Embedding Model API",
    similar_methods=["EmbeddingModel.embed_documents", "DocumentEmbedding.embed_documents"],
    common_parameters=["documents", "batch_size", "show_progress", "device"],
    common_return_type="numpy.ndarray",
    example_code="""
# All embedding methods follow the same pattern:
embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(
    documents=sample_docs,  # List[str] or pandas.Series
    batch_size=32,          # Process in batches
    show_progress=True      # Show progress bar
)
# Returns: numpy.ndarray of shape (n_documents, embedding_dimension)
"""
)

# 2. Document consistent topic modeling interface
document_consistent_api(
    method_name="Topic Model API",
    similar_methods=["BERTopicModel.fit", "Top2VecModel.fit", "UnifiedTopicModeler.fit"],
    common_parameters=["documents", "embeddings", "num_topics/n_topics"],
    common_return_type="self (for method chaining)",
    example_code="""
# All topic modeling methods follow the same pattern:
topic_model = UnifiedTopicModeler(method="embedding_cluster")
# Option 1: Pass documents and compute embeddings internally
topic_model.fit(
    documents=sample_docs,
    num_topics=5           # Parameter name standardized across all methods
)
# Option 2: Pass pre-computed embeddings
topic_model.fit(
    documents=sample_docs,
    embeddings=precomputed_embeddings,
    num_topics=5
)
# All .fit() methods return self for method chaining:
topic_model.fit(...).transform(...).visualize_topics(...)
"""
)

# 3. Document consistent transform interface
document_consistent_api(
    method_name="Transform API",
    similar_methods=["BERTopicModel.transform", "Top2VecModel.transform", "UnifiedTopicModeler.transform"],
    common_parameters=["documents", "embeddings"],
    common_return_type="Tuple[List[int], numpy.ndarray]",
    example_code="""
# All transform methods follow the same pattern:
# First, fit the model
topic_model.fit(documents=sample_docs)

# Then transform new documents
topics, probs = topic_model.transform(
    documents=new_docs,
    embeddings=None  # Optional, will compute if not provided
)
# Returns: Tuple of (topic_assignments, probabilities)
# - topic_assignments: List[int] with topic ID for each document
# - probabilities: numpy.ndarray with probability matrix
"""
)

# 4. Document consistent visualization parameter naming
document_consistent_api(
    method_name="Visualization API",
    similar_methods=[
        "visualize_topics", "visualize_embeddings", 
        "create_topic_distribution_plot", "create_embedding_visualization"
    ],
    common_parameters=[
        "width", "height", "title", "return_fig", 
        "colorscale", "marker_size"
    ],
    common_return_type="plotly.graph_objects.Figure",
    example_code="""
# All visualization functions use consistent parameter names:
fig = topic_model.visualize_topics(
    width=800,              # Consistent across all plots
    height=600,             # Consistent across all plots
    title="My Topics",      # Consistent across all plots
    return_fig=True,        # All visualizations can return or show
    colorscale="Viridis"    # Consistent color options
)

# Sample of consistent API parameters for different visualization types:
from meno.visualization.umap_viz import create_embedding_visualization
fig = create_embedding_visualization(
    embeddings=model.document_embeddings,
    labels=model.topic_assignments,
    width=800,              # Same parameter name and meaning
    height=600,             # Same parameter name and meaning
    title="Embeddings",     # Same parameter name and meaning
    marker_size=5,          # Consistent parameter name
    return_fig=True         # Consistent return behavior
)
"""
)

# 5. Document consistent model creation and configuration
document_consistent_api(
    method_name="Model Creation API",
    similar_methods=[
        "create_topic_modeler", "create_embedding_model"
    ],
    common_parameters=["method/model_name", "config_overrides"],
    common_return_type="Model instance (BERTopicModel, Top2VecModel, etc.)",
    example_code="""
# All model factory functions use consistent patterns:
from meno.modeling.unified_topic_modeling import create_topic_modeler

# Create a topic modeler with default settings
model = create_topic_modeler(method="bertopic")

# Override configuration with consistent dictionary structure
model = create_topic_modeler(
    method="bertopic",
    config_overrides={
        "n_topics": 10,
        "min_topic_size": 5,
        "embedding_model": "all-MiniLM-L6-v2"
    }
)

# Same pattern for embedding models
from meno.modeling.embeddings import create_embedding_model
embedding_model = create_embedding_model(
    model_name="all-MiniLM-L6-v2",
    config_overrides={
        "device": "cpu",
        "quantize": True
    }
)
"""
)

# 6. Document consistent workflow API
document_consistent_api(
    method_name="Workflow API",
    similar_methods=[
        "load_data", "preprocess_documents", "discover_topics", 
        "generate_acronym_report", "generate_misspelling_report"
    ],
    common_parameters=[
        "data/text_column", "output_path", "open_browser"
    ],
    common_return_type="Output path as string or DataFrame",
    example_code="""
# All workflow methods follow consistent patterns:
workflow = MenoWorkflow()

# Load data with consistent column specification
workflow.load_data(
    data=df,                # DataFrame input
    text_column="text",     # Name for the text column used consistently
    id_column="id",         # Consistent parameter names
    category_column="category"
)

# Generate reports with consistent parameters
acronym_report = workflow.generate_acronym_report(
    min_count=2,
    output_path="acronyms.html",  # Consistent path parameter
    open_browser=True             # Consistent behavior parameter
)

# Topic discovery with consistent parameter names
topics_df = workflow.discover_topics(
    method="embedding_cluster",   # Consistent with create_topic_modeler
    num_topics=5                 # Consistent parameter name
)
"""
)

# Try to run a practical demonstration if dependencies are available
try:
    print("\n🔬 Running practical API demonstration...")
    
    # Create embedding model
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    # Create topic model
    topic_model = create_topic_modeler(method="embedding_cluster")
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(
        documents=sample_docs,
        batch_size=5,
        show_progress=False
    )
    
    # Fit topic model
    topics = topic_model.fit_transform(
        documents=sample_docs,
        embeddings=embeddings,
        num_topics=3
    )
    
    # Visualize topics (use consistent visualization parameters)
    try:
        fig = topic_model.visualize_topics(
            width=800,
            height=600,
            title="ML & DL Topics", 
            return_fig=True
        )
        
        # Save visualization
        output_path = output_dir / "topic_visualization.html"
        fig.write_html(str(output_path))
        print(f"Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Visualization step skipped: {str(e)}")
    
    # Demonstrate workflow with consistent API
    print("\nDemonstrating workflow with consistent API...")
    workflow = MenoWorkflow()
    
    # Load data with consistent parameter names
    workflow.load_data(
        data=sample_df,
        text_column="text",
        category_column="category"
    )
    
    # Preprocess with consistent parameter names
    workflow.preprocess_documents(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True
    )
    
    # Discover topics with consistent parameter names
    topics_df = workflow.discover_topics(
        method="embedding_cluster",
        num_topics=3
    )
    
    # Get topic assignments
    print("\nTopic assignments for sample documents:")
    print(topics_df[["topic"]].head(3))
    
    # Generate report with consistent parameter names
    report_path = workflow.generate_comprehensive_report(
        output_path=output_dir / "workflow_report.html",
        title="API Consistency Demo",
        include_interactive=True,
        open_browser=False
    )
    print(f"Report generated at {report_path}")
    
except Exception as e:
    print(f"\n⚠️ Practical demonstration error: {str(e)}")
    print("Some features may require additional dependencies.")
    print("Install full dependencies with: pip install meno[full]")

print("\n✅ API Consistency Demonstration Complete")
print("""
The Meno library maintains consistent API patterns across components:

1. Parameter Naming: Same parameters have the same names and behavior
2. Return Types: Similar methods return consistent types
3. Method Naming: Related functionality uses predictable method names
4. Configuration: Consistent configuration hierarchy and overrides
5. Visualization: Standardized parameters for all visualizations

These consistent patterns make the library easier to learn and use,
reducing the cognitive load when switching between components.
""")