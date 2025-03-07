# Meno Web Interface

The Meno web interface provides an interactive, browser-based GUI for topic modeling without writing code. It's designed to make topic modeling accessible to users of all technical backgrounds.

![Meno Web Interface](../images/web_interface_screenshot.png)

## Features

- **Interactive Data Upload**: Upload CSV or text files directly through the browser
- **Model Configuration**: Configure topic models through an intuitive interface
- **Interactive Visualizations**: Explore topics with dynamic, interactive visualizations
- **Document Search**: Search and explore documents by topic or content
- **No-Code Experience**: Perform complete topic modeling workflow without writing code

## Getting Started

### Installation

The web interface is included in the core Meno package. Make sure you have the required dependencies:

```bash
pip install "meno[web]"
```

### Launching the Interface

You can launch the web interface from the command line:

```bash
meno-web --port 8050 --debug
```

Or from Python:

```python
from meno.web_interface import launch_web_interface

launch_web_interface(port=8050, debug=True)
```

This will start a local web server and open the interface in your default browser.

## Usage Guide

### 1. Data Upload

The web interface accepts data in several formats:

- **CSV files**: Files with text columns
- **TXT files**: Plain text files with one document per line
- **Direct text input**: For quick experiments with a small number of documents

To upload data:

1. Navigate to the "Data Upload" tab
2. Either:
   - Drag and drop a CSV/TXT file onto the upload area
   - Click the upload area to select a file
   - Enter text directly in the sample text area

After upload, you'll see a preview of the documents and basic statistics about your dataset.

### 2. Model Configuration

After uploading data, you can configure your topic model:

1. Navigate to the "Configure Model" tab
2. Select the model type:
   - **Simple K-Means**: Embedding-based clustering (balanced performance)
   - **TF-IDF K-Means**: Word frequency-based clustering (fastest)
   - **NMF Topic Model**: Matrix factorization (best interpretability)
   - **LSA Topic Model**: Latent semantic analysis (good for relationships)
3. Set the number of topics (typically 5-20)
4. Configure advanced options (optional):
   - Random seed for reproducibility
   - Maximum features for vocabulary size
   - Preprocessing options (stopwords, lowercasing, lemmatization)
5. Click "Train Model" to start the modeling process

The model description provides guidance on when to use each model type.

### 3. Exploring Results

After model training, the results tab provides multiple ways to explore your topics:

#### Topics Overview

- **Topic Distribution**: Bar chart showing number of documents per topic
- **Topic Table**: List of topics with names and top keywords
- **Interactive Selection**: Click any topic to see more details

#### Topic Landscape

An interactive 2D visualization showing relationships between topics and their relative positions in the semantic space.

- **Topic Clusters**: Topics are represented as nodes
- **Connections**: Lines between related topics
- **Size**: Node size indicates the number of documents in each topic

#### Topic Details

When selecting a specific topic, you'll see:

- **Top Words**: Key terms defining the topic with their weights
- **Sample Documents**: Representative documents from this topic
- **Word Distribution**: Visualization of word importance

#### Document-Topic Analysis

A heatmap visualization showing how documents relate to different topics:

- **Rows**: Individual documents
- **Columns**: Topics
- **Color Intensity**: Strength of association between document and topic
- **Configuration**: Adjust the number of documents displayed using the slider

### 4. Document Search

The search tab allows you to find specific documents:

1. Enter a search query
2. Optionally filter by topic
3. Choose sorting method (relevance or topic score)

Search results show:
- Document text
- Associated topic
- Relevance score

## Customization

### Programmatic Interface

You can create and customize the web app programmatically:

```python
from meno.web_interface import MenoWebApp

# Create custom instance
app = MenoWebApp(port=9000, debug=True)

# Access the underlying Dash app
dash_app = app.app

# Add custom callbacks or components
@dash_app.callback(...)
def custom_callback(...):
    ...

# Run the app
app.run()

# Clean up when done
app.cleanup()
```

### Integration with Existing Dash Apps

You can integrate the Meno web interface into an existing Dash application:

```python
import dash
from dash import html
from meno.web_interface import MenoWebApp

# Create your main Dash app
app = dash.Dash(__name__)

# Create Meno web app
meno_app = MenoWebApp()
meno_layout = meno_app.app.layout

# Integrate into your layout
app.layout = html.Div([
    html.H1("My Custom Dashboard"),
    html.Div([
        meno_layout
    ], className="meno-container")
])

# Run your app
if __name__ == "__main__":
    app.run_server(debug=True)
```

## Case Studies

### Text Analysis for Customer Feedback

A retail company used the Meno web interface to analyze customer feedback data:

1. Uploaded 5,000 customer comments from a CSV file
2. Used the NMF Topic Model with 8 topics
3. Discovered key themes in customer feedback:
   - Product quality issues
   - Shipping delays
   - Positive customer service experiences
   - Website usability problems
4. Identified actionable insights through document search
5. Exported findings to present to management

### Academic Research Paper Analysis

A research team used the web interface to analyze scientific abstracts:

1. Uploaded 1,200 research paper abstracts as a text file
2. Used the LSA Topic Model with 12 topics
3. Identified emerging research directions
4. Used the document search to find papers relevant to specific topics
5. Exported visualizations for publication

## Advanced Topics

### Resource Utilization

The web interface is designed to be lightweight:

- **CPU-first**: All modeling runs on CPU by default
- **Memory Efficient**: Streaming processing for large files
- **Lightweight Models**: Uses efficient alternatives to BERTopic

Typical resource requirements:
- 1-2 CPU cores
- 1-4GB RAM depending on dataset size
- No GPU required

### Security Considerations

The web interface runs locally on your machine:

- No data is sent to external servers
- Files are processed in a temporary directory
- Temporary files are cleaned up on exit

For multi-user deployments, consider:
- Running behind a reverse proxy
- Adding authentication
- Containerizing with Docker

### Limitations

Current limitations of the web interface:

- Maximum recommended dataset size: ~100,000 documents
- No persistence between sessions (results not saved)
- Limited customization of visualization appearance
- No export functionality for models (use Python API for this)

## Troubleshooting

### Common Issues

**Interface doesn't load:**
- Check that the port is not in use
- Ensure you have all required dependencies
- Try a different browser

**Error during data upload:**
- Ensure CSV files have a text column
- Check for encoding issues (use UTF-8)
- Try with a smaller sample of data

**Model training fails:**
- Reduce the number of topics
- Try a different model type
- Check for invalid characters in the text

**Visualizations don't appear:**
- Check your browser's JavaScript settings
- Try a different browser
- Reduce the number of documents displayed

### Getting Help

If you encounter issues:
- Check the console output for error messages
- Visit the [GitHub issues page](https://github.com/yourusername/meno/issues)
- Join the community discussion forum