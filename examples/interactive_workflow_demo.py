"""
Interactive Workflow Demo for Meno Topic Modeling Toolkit

This example demonstrates the interactive workflow for:
1. Acronym detection and expansion
2. Spelling correction
3. Topic modeling
4. Visualization

The workflow includes interactive reports that allow users to review
and customize acronym expansions and spelling corrections before proceeding
with topic modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Add the parent directory to the path so we can import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meno import MenoWorkflow

# Create a simple dataset with acronyms and misspellings
data = pd.DataFrame({
    "text": [
        "The CEO and CFO met to discuss the AI implementation in our CRM system.",
        "HR dept is implementing a new PTO policy next month.",
        "IT team resolved the API issue affecting the CX system.",
        "Customer submitted a claim for their vehical accident on HWY 101.",
        "The CTO presented the ML strategy for improving cust retention.",
        "CSRs are required to document all NPS feedback from customers.",
        "Policyholder recieved the EOB and was confused about the CPT codes.",
        "The investgation team finished the SIU referral for the claim.",
        "QA team found a bug in the OCR system that processes medcial forms.",
        "The underwritter rejected the application due to insuficient data.",
        "ROI for the project was calculated by the acounting department.",
        "The payment was delayed due to incorect ACH information."
    ],
    "date": pd.date_range(start="2023-01-01", periods=12, freq="W"),
    "department": ["Executive", "HR", "IT", "Claims", "Technology", 
                  "Customer Service", "Claims", "Investigation", "Quality", 
                  "Underwriting", "Finance", "Billing"],
    "region": ["North", "South", "East", "West", "North", "South", 
              "East", "West", "North", "South", "East", "West"]
})

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Initialize the workflow
print("Initializing the MenoWorkflow...")
workflow = MenoWorkflow()

# Load the data
print("Loading data...")
workflow.load_data(
    data=data,
    text_column="text",
    time_column="date",
    category_column="department",
    geo_column="region"
)

# Detect and report acronyms
print("Detecting acronyms...")
acronym_report_path = workflow.generate_acronym_report(
    min_length=2,
    min_count=1,  # Set to 1 for our small example dataset
    output_path=str(output_dir / "acronym_report.html"),
    open_browser=False  # Set to True to automatically open in browser
)
print(f"Acronym report generated at {acronym_report_path}")

# Based on the report, let's define custom acronym mappings
custom_acronym_mappings = {
    "CRM": "Customer Relationship Management",
    "PTO": "Paid Time Off",
    "CX": "Customer Experience",
    "HWY": "Highway",
    "CSR": "Customer Service Representative",
    "NPS": "Net Promoter Score",
    "EOB": "Explanation of Benefits",
    "CPT": "Current Procedural Terminology",
    "SIU": "Special Investigation Unit",
    "OCR": "Optical Character Recognition",
    "ACH": "Automated Clearing House"
}

# Expand acronyms with our custom mappings
print("Expanding acronyms with custom mappings...")
workflow.expand_acronyms(custom_mappings=custom_acronym_mappings)

# Detect and report potential misspellings
print("Detecting potential misspellings...")
misspelling_report_path = workflow.generate_misspelling_report(
    min_length=3,
    min_count=1,  # Set to 1 for our small example dataset
    output_path=str(output_dir / "misspelling_report.html"),
    open_browser=False  # Set to True to automatically open in browser
)
print(f"Misspelling report generated at {misspelling_report_path}")

# Based on the report, define custom spelling corrections
custom_spelling_corrections = {
    "vehical": "vehicle",
    "cust": "customer",
    "recieved": "received",
    "investgation": "investigation",
    "underwritter": "underwriter",
    "medcial": "medical",
    "insuficient": "insufficient",
    "acounting": "accounting",
    "incorect": "incorrect"
}

# Apply spelling corrections
print("Correcting spelling with custom corrections...")
workflow.correct_spelling(custom_corrections=custom_spelling_corrections)

# Preprocess the documents
print("Preprocessing documents...")
workflow.preprocess_documents(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemmatize=True
)

# Discover topics
print("Discovering topics...")
topics_df = workflow.discover_topics(
    method="embedding_cluster",
    num_topics=5,  # We'll create 5 topics for our small dataset
)
print(f"Discovered {len(topics_df['topic'].unique())} topics.")

# Visualize the topics with different visualizations
print("Generating visualizations...")

# Create embeddings visualization
embeddings_viz = workflow.visualize_topics(plot_type="embeddings")
embeddings_viz.write_html(str(output_dir / "topic_embeddings.html"))

# Create topic distribution visualization
distribution_viz = workflow.visualize_topics(plot_type="distribution")
distribution_viz.write_html(str(output_dir / "topic_distribution.html"))

# Create topic trends visualization (time-based)
trends_viz = workflow.visualize_topics(plot_type="trends")
trends_viz.write_html(str(output_dir / "topic_trends.html"))

# Generate a comprehensive report
print("Generating comprehensive report...")
report_path = workflow.generate_comprehensive_report(
    output_path=str(output_dir / "comprehensive_report.html"),
    open_browser=False,  # Set to True to automatically open in browser
    title="Interactive Workflow Demo Results",
    include_interactive=True,
    include_raw_data=True
)
print(f"Comprehensive report generated at {report_path}")
print("\nWorkflow complete! Check the output directory for results.")

print("""
Example Usage:
1. Run this script to generate reports
2. Review the acronym report and update custom_acronym_mappings as needed
3. Review the misspelling report and update custom_spelling_corrections as needed
4. Re-run with adjusted mappings and corrections if necessary
5. Examine the topic modeling results and visualizations
""")