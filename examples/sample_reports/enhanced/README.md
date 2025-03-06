# Enhanced HTML Reports in Meno

This directory contains sample HTML reports demonstrating the enhanced report generation capabilities in Meno. These reports showcase the improved visualizations, interactivity, and styling that are now available.

## Sample Reports

1. **[Comprehensive Report](comprehensive_report.html)** - Demonstrates all features including document embeddings, topic similarity, word clouds, and raw data export.

2. **[Similarity-Focused Report](similarity_focused_report.html)** - Focuses on topic relationships with the similarity heatmap.

3. **[Word Cloud-Focused Report](wordcloud_focused_report.html)** - Highlights the interactive word clouds for exploring topic content.

4. **[MenoTopicModeler Report](modeler_enhanced_report.html)** - Generated using the MenoTopicModeler class, showing how to use the enhanced reporting in your normal workflow.

## Key Improvements

- **Modern, Card-Based Design**: Clean, responsive interface with improved typography and visual organization
- **Interactive Tabs**: Navigate between different visualizations in a single view
- **Topic Similarity Analysis**: Heatmap showing relationships between topics
- **Interactive Word Clouds**: Visualize and explore top words for each topic
- **Data Export**: Export raw topic data to CSV with one click
- **Summary Metrics**: Clear dashboard with key statistics
- **Responsive Design**: Reports look great on desktop, tablet, or mobile

## How to Generate Enhanced Reports

### Using MenoTopicModeler

```python
from meno.meno import MenoTopicModeler

# Initialize and run topic modeling
modeler = MenoTopicModeler()
# ... process data and discover topics ...

# Generate enhanced report
report_path = modeler.generate_report(
    output_path="my_enhanced_report.html",
    title="My Topic Analysis",
    include_interactive=True,
    include_raw_data=True,
    max_examples_per_topic=5,
    similarity_matrix=my_similarity_matrix,  # Optional
    topic_words=my_topic_words  # Optional
)
```

### Using the HTML Generator Directly

```python
from meno.reporting.html_generator import generate_html_report

# After you have your data ready
report_path = generate_html_report(
    documents=documents_df,
    topic_assignments=topic_assignments_df,
    umap_projection=embeddings,
    output_path="my_report.html",
    config={
        "title": "My Custom Report",
        "include_interactive": True,
        "max_examples_per_topic": 5,
        "include_raw_data": True,
    },
    similarity_matrix=similarity_matrix,  # Optional
    topic_words=topic_words  # Optional
)
```

## Code Details

These reports were generated using the script `examples/generate_enhanced_report.py`. To see how the data was prepared and how the reports were configured, check out this script.

The key improvements to the HTML report generation include:

1. **HTML Template Enhancements**: Modernized design with CSS variables, responsive grids, and interactive elements
2. **New Visualization Types**: Added similarity heatmap and interactive word clouds
3. **Tab-Based Navigation**: Organized visualizations into tabs for better user experience
4. **Export Functionality**: Added JS-based CSV export for data tables
5. **Compatibility Fix**: Proper handling of Pydantic configuration models