"""HTML report generation utilities."""

from typing import Dict, List, Optional, Union, Any
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import jinja2
import plotly.graph_objects as go

from ..visualization.static_plots import plot_topic_distribution
from ..visualization.interactive_plots import plot_embeddings


# Simple HTML template for reports
DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .topic-examples {
            margin-bottom: 30px;
        }
        .example-text {
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 3px solid #2c3e50;
            margin-bottom: 10px;
        }
        .section {
            margin-bottom: 40px;
        }
        .visualization {
            margin: 20px 0;
        }
    </style>
    {{ plotly_js }}
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        <div class="section">
            <h2>Summary</h2>
            <p>
                Analysis of {{ document_count }} documents with {{ topic_count }} topics.
                Generated on {{ generation_date }}.
            </p>
        </div>

        <div class="section">
            <h2>Topic Distribution</h2>
            <div class="visualization">
                {{ topic_distribution_plot }}
            </div>
        </div>

        {% if umap_plot %}
        <div class="section">
            <h2>Document Embeddings</h2>
            <div class="visualization">
                {{ umap_plot }}
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>Topic Details</h2>
            {% for topic in topics %}
            <div class="topic-examples">
                <h3>{{ topic.name }} ({{ topic.count }} documents)</h3>
                
                {% if topic.top_words %}
                <p><strong>Top words:</strong> {{ topic.top_words }}</p>
                {% endif %}
                
                <h4>Example Documents:</h4>
                {% for example in topic.examples %}
                <div class="example-text">{{ example }}</div>
                {% endfor %}
            </div>
            {% endfor %}
        </div>

        {% if include_raw_data %}
        <div class="section">
            <h2>Raw Data</h2>
            <div id="topic-assignments-table">
                {{ topic_assignments_table }}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def generate_html_report(
    documents: pd.DataFrame,
    topic_assignments: pd.DataFrame,
    umap_projection: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    template: Optional[str] = None,
) -> str:
    """Generate an HTML report of topic modeling results.
    
    Parameters
    ----------
    documents : pd.DataFrame
        DataFrame with document texts and topic assignments
    topic_assignments : pd.DataFrame
        DataFrame with topic assignments and probabilities
    umap_projection : Optional[np.ndarray], optional
        UMAP projection for visualization, by default None
    output_path : Optional[Union[str, Path]], optional
        Path to save the report, by default None
        If None, creates a file in the current directory
    config : Optional[Dict[str, Any]], optional
        Report configuration, by default None
    template : Optional[str], optional
        Custom Jinja2 template, by default None
    
    Returns
    -------
    str
        Path to the generated report
    """
    # Default configuration
    default_config = {
        "title": "Topic Modeling Results",
        "include_interactive": True,
        "max_examples_per_topic": 5,
        "include_raw_data": False,
    }
    
    # Merge with provided config
    config = {**default_config, **(config or {})}
    
    # Use default template if not provided
    if template is None:
        template = DEFAULT_TEMPLATE
    
    # Create output path if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"topic_modeling_report_{timestamp}.html"
    
    # Convert to Path object
    output_path = Path(output_path)
    
    # Create plotly JS header for interactive visualizations
    plotly_js = ""
    if config["include_interactive"]:
        plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    
    # Create topic distribution plot
    topic_distribution_fig = plot_topic_distribution(
        documents["topic"],
        title="Topic Distribution",
    )
    topic_distribution_plot = topic_distribution_fig.to_html(
        full_html=False,
        include_plotlyjs=False,
    )
    
    # Create UMAP plot if projection provided
    umap_plot = None
    if umap_projection is not None and config["include_interactive"]:
        umap_fig = plot_embeddings(
            umap_projection,
            documents["topic"],
            document_texts=documents["text"],
            title="Document Embeddings",
        )
        umap_plot = umap_fig.to_html(
            full_html=False,
            include_plotlyjs=False,
        )
    
    # Get topic information
    topics_info = []
    for topic_name, topic_group in documents.groupby("topic"):
        # Skip if no documents
        if len(topic_group) == 0:
            continue
        
        # Get top words if available
        top_words = ""
        if "top_words" in topic_assignments.columns:
            top_words_row = topic_assignments[
                topic_assignments["topic"] == topic_name
            ]["top_words"].iloc[0]
            if isinstance(top_words_row, str):
                top_words = top_words_row
        
        # Get example documents
        examples = topic_group["text"].head(config["max_examples_per_topic"]).tolist()
        
        # Add topic info
        topics_info.append({
            "name": topic_name,
            "count": len(topic_group),
            "top_words": top_words,
            "examples": examples,
        })
    
    # Sort topics by count
    topics_info.sort(key=lambda x: x["count"], reverse=True)
    
    # Create HTML table for raw data
    topic_assignments_table = ""
    if config["include_raw_data"]:
        # Select subset of columns for display
        display_cols = ["topic", "topic_probability"]
        
        # Add topic similarity columns if available
        similarity_cols = [
            col for col in topic_assignments.columns
            if col.endswith("_similarity")
        ]
        display_cols.extend(similarity_cols)
        
        # Create HTML table
        table_df = pd.concat(
            [documents[["text"]], topic_assignments[display_cols]],
            axis=1,
        )
        topic_assignments_table = table_df.to_html(
            index=False,
            classes="dataframe",
            float_format="%.3f",
        )
    
    # Create context for template
    from datetime import datetime
    context = {
        "title": config["title"],
        "document_count": len(documents),
        "topic_count": len(topics_info),
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plotly_js": plotly_js,
        "topic_distribution_plot": topic_distribution_plot,
        "umap_plot": umap_plot,
        "topics": topics_info,
        "include_raw_data": config["include_raw_data"],
        "topic_assignments_table": topic_assignments_table,
    }
    
    # Render template
    html = jinja2.Template(template).render(**context)
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return str(output_path)


def generate_json_report(
    documents: pd.DataFrame,
    topic_assignments: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a JSON report of topic modeling results.
    
    Parameters
    ----------
    documents : pd.DataFrame
        DataFrame with document texts and topic assignments
    topic_assignments : pd.DataFrame
        DataFrame with topic assignments and probabilities
    output_path : Optional[Union[str, Path]], optional
        Path to save the report, by default None
        If None, creates a file in the current directory
    
    Returns
    -------
    str
        Path to the generated report
    """
    # Create output path if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"topic_modeling_report_{timestamp}.json"
    
    # Convert to Path object
    output_path = Path(output_path)
    
    # Prepare data for JSON
    from datetime import datetime
    
    # Get topic information
    topics_info = []
    for topic_name, topic_group in documents.groupby("topic"):
        # Get example documents
        examples = topic_group["text"].head(5).tolist()
        
        # Add topic info
        topics_info.append({
            "name": topic_name,
            "count": int(len(topic_group)),
            "examples": examples,
        })
    
    # Create report data
    report_data = {
        "metadata": {
            "document_count": len(documents),
            "topic_count": len(topics_info),
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "topics": topics_info,
        "topic_distribution": documents["topic"].value_counts().to_dict(),
    }
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    return str(output_path)