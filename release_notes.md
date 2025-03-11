# Meno v1.1.1: Feedback Visualization System

This release adds specialized visualization tools for analyzing the impact of user feedback on topic models.

## New Features
- **Feedback Impact Visualization**: Comprehensive dashboard showing how feedback affects topic distributions
- **Topic Transition Analysis**: Visual representation of how documents move between topics
- **Interactive Comparison Dashboard**: Web-based interactive tool for exploring before/after feedback changes
- **Session Progress Tracking**: Visualize the cumulative impact of feedback sessions

## New Files
- Added meno/visualization/enhanced_viz/feedback_viz.py module with visualization functions
- Added examples/feedback_visualization_example.py demonstration script
- Added examples/feedback_visualization_notebook.ipynb interactive demo notebook

## API Changes
- Added top-level exports for visualization functions:
  - plot_feedback_impact
  - create_feedback_comparison_dashboard
  - plot_topic_feedback_distribution
  
## Documentation
- Updated README with examples of the new feedback visualization capabilities
- Added the visualization system to the list of new features in v1.1
