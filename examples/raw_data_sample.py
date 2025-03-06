#!/usr/bin/env python
"""
Generate a sample report demonstrating the improved raw data handling.
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

from meno.meno import MenoTopicModeler
from meno.reporting.html_generator import generate_html_report

# Output directory
OUTPUT_DIR = Path("examples/sample_reports/demo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a larger sample dataset
def generate_sample_data(n_samples=500):
    """Generate a larger synthetic dataset."""
    topics = ["Technology", "Healthcare", "Finance", "Education", "Environment"]
    
    # Generate documents
    data = []
    for i in range(n_samples):
        topic = random.choice(topics)
        
        if topic == "Technology":
            text = f"Sample {i}: Discussion about technology innovations including {random.choice(['AI', 'blockchain', 'IoT', 'cloud computing'])}."
        elif topic == "Healthcare":
            text = f"Sample {i}: Analysis of healthcare trends in {random.choice(['telemedicine', 'preventive care', 'medical research', 'patient data'])}"
        elif topic == "Finance":
            text = f"Sample {i}: Financial report on {random.choice(['market trends', 'investment strategies', 'economic indicators', 'banking regulations'])}"
        elif topic == "Education":
            text = f"Sample {i}: Education system review focusing on {random.choice(['online learning', 'curriculum development', 'student assessment', 'teacher training'])}"
        else:  # Environment
            text = f"Sample {i}: Environmental study about {random.choice(['climate change', 'renewable energy', 'conservation', 'sustainable development'])}"
        
        data.append({
            "text": text,
            "topic": topic,
            "topic_probability": random.uniform(0.7, 0.99)
        })
    
    return pd.DataFrame(data)

# Generate sample data
print("Generating sample data...")
df = generate_sample_data(n_samples=500)

# Create a typical topic_assignments DataFrame
topic_assignments = pd.DataFrame({
    "topic": df["topic"],
    "topic_probability": df["topic_probability"]
})

# Add some similarity columns for demonstration
for topic in df["topic"].unique():
    topic_assignments[f"{topic}_similarity"] = np.where(
        df["topic"] == topic,
        df["topic_probability"],
        np.random.uniform(0.01, 0.3, size=len(df))
    )

# Generate a report with the full raw data table (for comparison)
print("Generating report with full raw data table...")
report_path = generate_html_report(
    documents=df,
    topic_assignments=topic_assignments,
    output_path=OUTPUT_DIR / "report_with_full_data.html",
    config={
        "title": "Topic Analysis - Full Raw Data",
        "include_interactive": True,
        "include_raw_data": True,
        "max_examples_per_topic": 5,
        "max_samples_per_topic": 9999  # Effectively all data
    }
)
print(f"Generated report with full raw data at {report_path}")

# Generate a report with sampled raw data table
print("Generating report with sampled raw data table...")
report_path = generate_html_report(
    documents=df,
    topic_assignments=topic_assignments,
    output_path=OUTPUT_DIR / "report_with_sampled_data.html",
    config={
        "title": "Topic Analysis - Sampled Raw Data",
        "include_interactive": True,
        "include_raw_data": True,
        "max_examples_per_topic": 5,
        "max_samples_per_topic": 5  # Only 5 examples per topic
    }
)
print(f"Generated report with sampled raw data at {report_path}")

# Create an index file to compare the reports
with open(OUTPUT_DIR / "index.html", "w") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raw Data Handling Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Roboto, -apple-system, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .comparison {
            display: flex;
            gap: 20px;
            margin-top: 30px;
        }
        .report-link {
            flex: 1;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .btn {
            display: inline-block;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: 600;
            margin-top: 10px;
        }
        .note {
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 10px 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Raw Data Handling Comparison</h1>
        
        <div class="note">
            <p>This demonstrates the improved raw data handling in HTML reports. The sampled version shows only 5 examples per topic, while still allowing export of the full dataset via CSV.</p>
        </div>
        
        <div class="comparison">
            <div class="report-link">
                <h3>Full Raw Data Report</h3>
                <p>Shows all 500 rows in the raw data table, which can be slow to load and navigate.</p>
                <a href="report_with_full_data.html" class="btn" target="_blank">View Report</a>
            </div>
            
            <div class="report-link">
                <h3>Sampled Raw Data Report</h3>
                <p>Shows only 5 examples per topic (25 total rows), but allows export of all 500 rows via CSV.</p>
                <a href="report_with_sampled_data.html" class="btn" target="_blank">View Report</a>
            </div>
        </div>
    </div>
</body>
</html>
""")

print(f"\nAll files generated in {OUTPUT_DIR}")
print("View the comparison at examples/sample_reports/demo/index.html")