"""HTML report generation utilities."""

from typing import Dict, List, Optional, Union, Any
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import jinja2
import plotly.graph_objects as go

from ..visualization.static_plots import plot_topic_distribution, plot_topic_word_clouds
from ..visualization.interactive_plots import (
    plot_embeddings, 
    plot_topic_clusters,
    plot_topic_similarity_heatmap,
    plot_interactive_wordcloud
)


# Enhanced HTML template for reports with modern styling and interactive features
DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #9b59b6;
            --light-bg: #f8f9fa;
            --dark-bg: #2c3e50;
            --text-color: #333;
            --light-text: #f8f9fa;
            --border-color: #e9ecef;
            --topic-colors: #e74c3c, #3498db, #2ecc71, #f39c12, #9b59b6, #1abc9c;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: var(--text-color);
            background-color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        header {
            background-color: var(--dark-bg);
            color: var(--light-text);
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        header .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-meta {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        h1, h2, h3, h4 {
            color: var(--secondary-color);
            font-weight: 600;
            margin-top: 0;
        }
        
        h1 {
            font-size: 2.2rem;
            color: var(--light-text);
            margin: 0;
        }
        
        h2 {
            font-size: 1.8rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h3 {
            font-size: 1.4rem;
        }
        
        .card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: var(--light-bg);
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .card-body {
            padding: 20px;
        }
        
        .visualization {
            margin: 20px 0;
            width: 100%;
            overflow: hidden;
        }
        
        .topic-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .topic-card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        
        .topic-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .topic-header {
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .topic-name {
            font-weight: 600;
            font-size: 1.2rem;
            margin: 0;
        }
        
        .topic-count {
            background-color: var(--primary-color);
            color: white;
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .topic-body {
            padding: 15px;
        }
        
        .topic-words {
            background-color: var(--light-bg);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: var(--secondary-color);
        }
        
        .example-text {
            background-color: var(--light-bg);
            padding: 10px 15px;
            border-left: 3px solid var(--primary-color);
            margin-bottom: 10px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 600;
            color: var(--text-color);
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        
        .tab.active {
            opacity: 1;
            border-bottom: 3px solid var(--primary-color);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--light-bg);
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background-color: var(--light-bg);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .summary-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .summary-label {
            font-size: 0.9rem;
            color: var(--secondary-color);
            margin-top: 5px;
        }
        
        .export-btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        
        .export-btn:hover {
            background-color: #2980b9;
        }
        
        .sample-note {
            background-color: #f8f9fa;
            border-left: 3px solid var(--primary-color);
            padding: 10px 15px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: var(--secondary-color);
            border-radius: 4px;
        }
        
        footer {
            background-color: var(--light-bg);
            padding: 20px 0;
            text-align: center;
            margin-top: 50px;
            font-size: 0.9rem;
            color: var(--secondary-color);
        }
        
        @media (max-width: 768px) {
            .topic-grid {
                grid-template-columns: 1fr;
            }
            
            .summary-cards {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
    {{ plotly_js }}
    <script>
        // Add interactive functionality after document loads
        document.addEventListener('DOMContentLoaded', function() {
            // Tab functionality
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab and corresponding content
                    tab.classList.add('active');
                    const contentId = tab.getAttribute('data-tab');
                    document.getElementById(contentId).classList.add('active');
                });
            });
            
            // Full CSV data for export
            const fullData = {{ full_data_json|safe }};
            
            // Export table to CSV
            const exportBtn = document.getElementById('export-csv');
            if (exportBtn) {
                exportBtn.addEventListener('click', () => {
                    if (!fullData || !fullData.columns || !fullData.data) {
                        // Fall back to exporting visible table if full data isn't available
                        exportVisibleTable();
                        return;
                    }
                    
                    // Create CSV from full dataset
                    let csv = [];
                    
                    // Add header row
                    csv.push(fullData.columns.map(col => `"${col}"`).join(','));
                    
                    // Add data rows
                    fullData.data.forEach(row => {
                        const rowData = row.map(cell => {
                            // Handle null values
                            if (cell === null) return '""';
                            // Format numbers to 3 decimal places
                            if (typeof cell === 'number') {
                                return `"${cell.toFixed(3)}"`;
                            }
                            // Escape quotes in text
                            return `"${String(cell).replace(/"/g, '""')}"`;
                        });
                        csv.push(rowData.join(','));
                    });
                    
                    downloadCSV(csv.join('\\n'));
                });
            }
            
            // Function to export only the visible table
            function exportVisibleTable() {
                const table = document.querySelector('.data-table');
                if (!table) return;
                
                let csv = [];
                const rows = table.querySelectorAll('tr');
                
                rows.forEach(row => {
                    const cols = row.querySelectorAll('td, th');
                    const rowData = Array.from(cols).map(col => '"' + col.innerText.replace(/"/g, '""') + '"');
                    csv.push(rowData.join(','));
                });
                
                downloadCSV(csv.join('\\n'));
            }
            
            // Helper function to download CSV data
            function downloadCSV(csvContent) {
                const blob = new Blob([csvContent], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.setAttribute('hidden', '');
                a.setAttribute('href', url);
                a.setAttribute('download', 'topic_data.csv');
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        });
    </script>
</head>
<body>
    <header>
        <div class="container">
            <h1>{{ title }}</h1>
            <div class="header-meta">
                Generated on {{ generation_date }}
            </div>
        </div>
    </header>

    <div class="container">
        <div class="card">
            <div class="card-header">
                <h2>Summary</h2>
            </div>
            <div class="card-body">
                <div class="summary-cards">
                    <div class="summary-card">
                        <div class="summary-value">{{ document_count }}</div>
                        <div class="summary-label">Documents</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value">{{ topic_count }}</div>
                        <div class="summary-label">Topics</div>
                    </div>
                    {% if avg_words_per_doc %}
                    <div class="summary-card">
                        <div class="summary-value">{{ avg_words_per_doc }}</div>
                        <div class="summary-label">Avg Words/Doc</div>
                    </div>
                    {% endif %}
                    {% if topic_coherence %}
                    <div class="summary-card">
                        <div class="summary-value">{{ topic_coherence }}</div>
                        <div class="summary-label">Topic Coherence</div>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Topic Distribution</h2>
            </div>
            <div class="card-body">
                <div class="visualization">
                    {{ topic_distribution_plot }}
                </div>
            </div>
        </div>

        {% if umap_plot %}
        <div class="card">
            <div class="card-header">
                <h2>Visualizations</h2>
            </div>
            <div class="card-body">
                <div class="tabs">
                    <div class="tab active" data-tab="embeddings-tab">Document Embeddings</div>
                    {% if topic_similarity_plot %}
                    <div class="tab" data-tab="similarity-tab">Topic Similarity</div>
                    {% endif %}
                    {% if topic_wordcloud_plot %}
                    <div class="tab" data-tab="wordcloud-tab">Word Clouds</div>
                    {% endif %}
                </div>
                
                <div id="embeddings-tab" class="tab-content active">
                    <div class="visualization">
                        {{ umap_plot }}
                    </div>
                </div>
                
                {% if topic_similarity_plot %}
                <div id="similarity-tab" class="tab-content">
                    <div class="visualization">
                        {{ topic_similarity_plot }}
                    </div>
                </div>
                {% endif %}
                
                {% if topic_wordcloud_plot %}
                <div id="wordcloud-tab" class="tab-content">
                    <div class="visualization">
                        {{ topic_wordcloud_plot }}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        <div class="card">
            <div class="card-header">
                <h2>Topic Details</h2>
            </div>
            <div class="card-body">
                <div class="topic-grid">
                    {% for topic in topics %}
                    <div class="topic-card">
                        <div class="topic-header">
                            <h3 class="topic-name">{{ topic.name }}</h3>
                            <span class="topic-count">{{ topic.count }}</span>
                        </div>
                        <div class="topic-body">
                            {% if topic.top_words %}
                            <div class="topic-words">
                                <strong>Top words:</strong> {{ topic.top_words }}
                            </div>
                            {% endif %}
                            
                            <h4>Example Documents:</h4>
                            {% for example in topic.examples %}
                            <div class="example-text">{{ example }}</div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        {% if include_raw_data %}
        <div class="card">
            <div class="card-header">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="margin: 0;">Raw Data</h2>
                    <button id="export-csv" class="export-btn">Export to CSV</button>
                </div>
            </div>
            <div class="card-body">
                <div id="topic-assignments-table">
                    {{ topic_assignments_table }}
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    
    <footer>
        <div class="container">
            Generated with Meno Topic Modeling Toolkit
        </div>
    </footer>
</body>
</html>
"""


def generate_html_report(
    documents: pd.DataFrame,
    topic_assignments: pd.DataFrame,
    umap_projection: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[Dict[str, Any], "HTMLReportConfig"]] = None,
    template: Optional[str] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    topic_words: Optional[Dict[str, Dict[str, float]]] = None,
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
    config : Optional[Union[Dict[str, Any], HTMLReportConfig]], optional
        Report configuration, by default None
    template : Optional[str], optional
        Custom Jinja2 template, by default None
    similarity_matrix : Optional[np.ndarray], optional
        Matrix of topic similarities, shape (n_topics, n_topics), by default None
    topic_words : Optional[Dict[str, Dict[str, float]]], optional
        Dictionary mapping topic names to word frequency dictionaries, by default None
    
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
    
    # Handle Pydantic model and convert to dict if needed
    if config is not None and hasattr(config, "dict"):
        config = config.dict()
    
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
        sort_by_count=True,
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
    
    # Create topic similarity heatmap if similarity matrix provided
    topic_similarity_plot = None
    if similarity_matrix is not None and config["include_interactive"]:
        topic_names = sorted(documents["topic"].unique())
        similarity_fig = plot_topic_similarity_heatmap(
            similarity_matrix,
            topic_names,
            title="Topic Similarity",
        )
        topic_similarity_plot = similarity_fig.to_html(
            full_html=False,
            include_plotlyjs=False,
        )
    
    # Create interactive word cloud if topic_words provided
    topic_wordcloud_plot = None
    if topic_words is not None and config["include_interactive"]:
        wordcloud_fig = plot_interactive_wordcloud(
            topic_words,
            title="Topic Word Clouds",
        )
        topic_wordcloud_plot = wordcloud_fig.to_html(
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
        elif topic_words and topic_name in topic_words:
            # Use top words from topic_words if available
            words = topic_words[topic_name]
            top_words = ", ".join(
                sorted(words.keys(), key=lambda k: words[k], reverse=True)[:10]
            )
        
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
    full_data_json = "{}"
    if config["include_raw_data"]:
        # Select subset of columns for display
        display_cols = ["topic", "topic_probability"]
        
        # Add topic similarity columns if available
        similarity_cols = [
            col for col in topic_assignments.columns
            if col.endswith("_similarity")
        ]
        display_cols.extend(similarity_cols)
        
        # Create a sample of the data by taking a few examples from each topic
        max_samples_per_topic = config.get("max_samples_per_topic", 5)
        
        # Get unique topics
        unique_topics = documents["topic"].unique()
        
        # Create sample DataFrame with examples from each topic
        sample_rows = []
        for topic_name in unique_topics:
            # Get rows for this topic, limited to max_samples_per_topic
            topic_rows = documents[documents["topic"] == topic_name].head(max_samples_per_topic)
            sample_rows.append(topic_rows)
        
        # Combine all sample rows
        sample_df = pd.concat(sample_rows)
        
        # Create HTML table
        table_df = pd.concat(
            [sample_df[["text"]], topic_assignments.loc[sample_df.index][display_cols]],
            axis=1,
        )
        
        # Add a note about the sampling
        sample_note = f"<div class='sample-note'><em>Note: Showing sample data ({max_samples_per_topic} examples per topic). Download CSV for full dataset.</em></div>"
        
        # Generate the HTML table
        topic_assignments_table = sample_note + table_df.to_html(
            index=False,
            classes="dataframe data-table",  # Add data-table class for CSV export
            float_format="%.3f",
        )
        
        # Create full data for CSV export
        full_table_df = pd.concat(
            [documents[["text"]], topic_assignments[display_cols]],
            axis=1,
        )
        
        # Convert to JSON for embedding in the HTML
        import json
        full_data_dict = {
            "columns": full_table_df.columns.tolist(),
            "data": full_table_df.values.tolist(),
        }
        full_data_json = json.dumps(full_data_dict)
    
    # Calculate additional metrics
    avg_words_per_doc = None
    if "text" in documents.columns:
        # Calculate average words per document
        avg_words = documents["text"].str.split().str.len().mean()
        avg_words_per_doc = f"{avg_words:.1f}"
    
    # Topic coherence score (placeholder - would be calculated by modeling)
    topic_coherence = None
    if "coherence" in topic_assignments.columns:
        topic_coherence = f"{topic_assignments['coherence'].mean():.2f}"
    
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
        "topic_similarity_plot": topic_similarity_plot,
        "topic_wordcloud_plot": topic_wordcloud_plot,
        "topics": topics_info,
        "include_raw_data": config["include_raw_data"],
        "topic_assignments_table": topic_assignments_table,
        "avg_words_per_doc": avg_words_per_doc,
        "topic_coherence": topic_coherence,
        "full_data_json": full_data_json,
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