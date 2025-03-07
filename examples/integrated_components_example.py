"""Example script demonstrating the integration of lightweight models, visualizations, and web interface.

This script shows how to use all the new components together, including:
1. Creating and training multiple lightweight models
2. Comparing models with visualizations
3. Using the web interface for interactive exploration
"""

import pandas as pd
import numpy as np
import plotly.io as pio
from pathlib import Path
import argparse
import sys

# Import lightweight models
from meno.modeling.simple_models.lightweight_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel, 
    LSATopicModel
)

# Import visualizations
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)

# Import web interface
# Commented out web interface import due to ModernTextEmbedding dependency
# from meno.web_interface import launch_web_interface


def load_sample_data():
    """Load sample data for demonstration."""
    # Example data - technology, healthcare, and environment topics
    sample_documents = [
        "Machine learning is a subfield of artificial intelligence that uses statistical techniques to enable computers to learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Neural networks are computing systems inspired by the biological neural networks in animal brains.",
        "Python is a popular programming language for data science and machine learning applications.",
        "TensorFlow and PyTorch are popular deep learning frameworks used to build neural networks.",
        "Natural language processing (NLP) enables computers to understand and interpret human language.",
        "Computer vision is a field of AI that enables computers to derive information from images and videos.",
        "Healthcare technology uses AI to improve diagnostics and patient care outcomes.",
        "Medical imaging uses computer vision techniques to analyze and interpret medical scans.",
        "Electronic health records (EHR) store patient data and medical history in digital format.",
        "Climate change refers to long-term shifts in global temperature and weather patterns.",
        "Renewable energy sources like solar and wind power help reduce carbon emissions.",
        "Sustainable development aims to meet human needs while preserving the environment.",
        "Conservation efforts focus on protecting biodiversity and natural habitats.",
        "Electric vehicles reduce reliance on fossil fuels and lower carbon emissions."
    ]
    
    # Create a dataframe with documents and IDs
    df = pd.DataFrame({
        "text": sample_documents,
        "doc_id": [f"doc_{i}" for i in range(len(sample_documents))]
    })
    
    print(f"Loaded {len(df)} sample documents")
    return df


def create_embedding_model_mock():
    """Create a mock embedding model."""
    import numpy as np
    
    class MockEmbeddingModel:
        def embed_documents(self, documents):
            """Generate random embeddings for documents."""
            print(f"    Generating embeddings for {len(documents)} documents...")
            return np.random.random((len(documents), 384))  # 384 is standard embedding dim
            
    return MockEmbeddingModel()

def create_and_fit_models(documents, num_topics=3):
    """Create and fit multiple lightweight topic models."""
    print("\nTraining lightweight topic models...")
    
    # Create models
    models = {
        "Simple K-Means": SimpleTopicModel(num_topics=num_topics, random_state=42),
        "TF-IDF K-Means": TFIDFTopicModel(num_topics=num_topics, random_state=42),
        "NMF Topic Model": NMFTopicModel(num_topics=num_topics, random_state=42),
        "LSA Topic Model": LSATopicModel(num_topics=num_topics, random_state=42)
    }
    
    # Create mock embedding model for SimpleTopicModel
    mock_embedder = create_embedding_model_mock()
    models["Simple K-Means"].embedding_model = mock_embedder
    
    # Fit models
    fitted_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(documents)
        fitted_models[name] = model
        
        # Print topic information
        topic_info = model.get_topic_info()
        print(f"    Found {len(topic_info)} topics:")
        for _, row in topic_info.iterrows():
            print(f"      Topic {row['Topic']}: {row['Name']}")
    
    return fitted_models


def create_visualizations(fitted_models, documents, output_dir=None):
    """Create and optionally save visualizations for model comparison."""
    print("\nCreating visualizations...")
    
    # Create output directory if provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Model comparison visualization
    print("  Creating model comparison visualization...")
    model_comparison_fig = plot_model_comparison(
        document_lists=[documents] * len(fitted_models),
        model_names=list(fitted_models.keys()),
        models=list(fitted_models.values())
    )
    
    # 2. Topic landscape visualizations
    print("  Creating topic landscape visualizations...")
    landscape_figs = {}
    for name, model in fitted_models.items():
        landscape_figs[name] = plot_topic_landscape(
            model=model,
            documents=documents,
            title=f"{name} Topic Landscape"
        )
    
    # 3. Multi-topic heatmap (comparing first two models)
    print("  Creating multi-topic comparison heatmap...")
    model_names = list(fitted_models.keys())
    model_list = list(fitted_models.values())
    heatmap_fig = plot_multi_topic_heatmap(
        models=model_list[:2],
        model_names=model_names[:2],
        document_lists=[documents] * 2
    )
    
    # 4. Document analysis for each model
    print("  Creating document analysis visualizations...")
    doc_analysis_figs = {}
    for name, model in fitted_models.items():
        doc_analysis_figs[name] = plot_comparative_document_analysis(
            model=model,
            documents=documents,
            title=f"{name} Document Analysis"
        )
    
    # Save visualizations if output directory is provided
    if output_dir:
        print(f"  Saving visualizations to {output_dir}...")
        pio.write_html(model_comparison_fig, str(output_path / "model_comparison.html"))
        
        for name, fig in landscape_figs.items():
            pio.write_html(fig, str(output_path / f"{name.lower().replace(' ', '_')}_landscape.html"))
        
        pio.write_html(heatmap_fig, str(output_path / "model_comparison_heatmap.html"))
        
        for name, fig in doc_analysis_figs.items():
            pio.write_html(fig, str(output_path / f"{name.lower().replace(' ', '_')}_doc_analysis.html"))
    
    return {
        "comparison": model_comparison_fig,
        "landscapes": landscape_figs,
        "heatmap": heatmap_fig,
        "doc_analysis": doc_analysis_figs
    }


def save_models(fitted_models, output_dir):
    """Save trained models to disk."""
    print(f"\nSaving models to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, model in fitted_models.items():
        model_dir = output_path / name.lower().replace(" ", "_")
        print(f"  Saving {name} to {model_dir}...")
        model.save(model_dir)
    
    print("All models saved successfully")


def launch_web_app(port=8050):
    """Launch the web interface for interactive exploration."""
    print(f"\nWeb interface feature is disabled in this test due to dependencies.")
    print("To use the web interface, install the full package with:")
    print("    pip install \"meno[web]\"")
    print("Then run this script with the --web flag.")


def main():
    """Main function to run the integrated example."""
    parser = argparse.ArgumentParser(description="Meno Lightweight Components Example")
    parser.add_argument("--topics", type=int, default=3, help="Number of topics to extract")
    parser.add_argument("--output", type=str, default="output/integrated_example", help="Output directory for visualizations and models")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--port", type=int, default=8050, help="Port for web interface")
    
    args = parser.parse_args()
    
    # 1. Load sample data
    print("=" * 80)
    print("MENO LIGHTWEIGHT COMPONENTS EXAMPLE")
    print("=" * 80)
    
    df = load_sample_data()
    documents = df["text"].tolist()
    
    # 2. Create and fit multiple models
    fitted_models = create_and_fit_models(documents, num_topics=args.topics)
    
    # 3. Create visualizations
    visualizations = create_visualizations(fitted_models, documents, args.output)
    
    # 4. Save models
    save_models(fitted_models, args.output)
    
    # 5. Launch web interface if requested
    if args.web:
        launch_web_app(args.port)
    else:
        print("\nTo explore models interactively, run this script with the --web flag")
    
    print("\nAll examples completed successfully!")
    print(f"Visualizations and models saved to {args.output}")


if __name__ == "__main__":
    main()