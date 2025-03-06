"""
Test script to run Meno workflow on a HuggingFace dataset with
artificial spatial, time, and category data to generate reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import random
from datetime import datetime, timedelta
import datasets

# Add the parent directory to the path so we can import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meno import MenoWorkflow

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Download and prepare the dataset
print("Downloading the Australian Insurance dataset from HuggingFace...")
dataset = datasets.load_dataset("answerai/australian-insurance-pii")
print(f"Dataset loaded: {len(dataset['train'])} samples")

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

# Prepare sample data - take a subset to make it faster
sample_size = 100
df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
print(f"Working with a sample of {len(df_sample)} records")

# Add synthetic time data - random dates within the last 3 years
start_date = datetime.now() - timedelta(days=3*365)
end_date = datetime.now()

def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

df_sample['date'] = [random_date(start_date, end_date) for _ in range(len(df_sample))]

# Add synthetic geographic data - Australian states and territories
states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
df_sample['state'] = np.random.choice(states, size=len(df_sample))

# Map states to GPS coordinates (approximate centroids)
state_coords = {
    'NSW': (-33.8688, 151.2093),  # Sydney
    'VIC': (-37.8136, 144.9631),  # Melbourne
    'QLD': (-27.4698, 153.0251),  # Brisbane
    'SA': (-34.9285, 138.6007),   # Adelaide
    'WA': (-31.9505, 115.8605),   # Perth
    'TAS': (-42.8821, 147.3272),  # Hobart
    'NT': (-12.4634, 130.8456),   # Darwin
    'ACT': (-35.2809, 149.1300)   # Canberra
}

df_sample['latitude'] = df_sample['state'].map(lambda s: state_coords[s][0])
df_sample['longitude'] = df_sample['state'].map(lambda s: state_coords[s][1])

# Add small random variations to coordinates to spread out the points
df_sample['latitude'] += np.random.normal(0, 0.5, len(df_sample))
df_sample['longitude'] += np.random.normal(0, 0.5, len(df_sample))

# Add synthetic category data - insurance categories
categories = ['Auto', 'Home', 'Health', 'Life', 'Business', 'Travel', 'Pet']
df_sample['category'] = np.random.choice(categories, size=len(df_sample))

# Add synthetic subcategories
subcategories = {
    'Auto': ['Collision', 'Comprehensive', 'Liability', 'Theft'],
    'Home': ['Building', 'Contents', 'Flood', 'Fire'],
    'Health': ['Hospital', 'Dental', 'Vision', 'Disability'],
    'Life': ['Term', 'Whole Life', 'Endowment', 'Group'],
    'Business': ['Property', 'Liability', 'Workers Comp', 'Interruption'],
    'Travel': ['Medical', 'Cancellation', 'Baggage', 'Delay'],
    'Pet': ['Accident', 'Illness', 'Dental', 'Preventative']
}

df_sample['subcategory'] = df_sample['category'].map(
    lambda c: np.random.choice(subcategories[c])
)

print("Created synthetic data fields:")
print("- Time data: 'date'")
print("- Geographic data: 'state', 'latitude', 'longitude'")
print("- Category data: 'category', 'subcategory'")

# Initialize the workflow
print("\nInitializing the MenoWorkflow...")
workflow = MenoWorkflow()

# Load the data
print("Loading data into workflow...")
workflow.load_data(
    data=df_sample,
    text_column="Text",
    time_column="date",
    geo_column="state",
    category_column="category"
)

# Generate acronym report
print("\nGenerating acronym report...")
acronym_report_path = workflow.generate_acronym_report(
    min_length=2,
    min_count=2,
    output_path=str(output_dir / "insurance_acronyms.html"),
    open_browser=False
)
print(f"Acronym report generated at {acronym_report_path}")

# Define custom acronym mappings from our analysis
custom_acronym_mappings = {
    "PDS": "Product Disclosure Statement",
    "CTP": "Compulsory Third Party",
    "CTP": "Compulsory Third Party",
    "PD": "Property Damage",
    "TD": "Total Disability",
    "TPD": "Total and Permanent Disability",
    "GIO": "Government Insurance Office",
    "AAMI": "Australian Associated Motor Insurers",
    "NRMA": "National Roads and Motorists' Association",
    "RACV": "Royal Automobile Club of Victoria",
    "RAC": "Royal Automobile Club",
    "CGU": "Commercial General Union",
    "QBE": "Queensland British Exporters",
    "CBA": "Commonwealth Bank of Australia"
}

# Expand acronyms with our custom mappings
print("Expanding acronyms with custom mappings...")
workflow.expand_acronyms(custom_mappings=custom_acronym_mappings)

# Generate spelling report
print("\nGenerating misspelling report...")
misspelling_report_path = workflow.generate_misspelling_report(
    min_length=4,
    min_count=2,
    output_path=str(output_dir / "insurance_misspellings.html"),
    open_browser=False
)
print(f"Misspelling report generated at {misspelling_report_path}")

# Define custom spelling corrections from our analysis
custom_spelling_corrections = {
    "policey": "policy",
    "insurnace": "insurance",
    "cliam": "claim",
    "premiume": "premium",
    "covrage": "coverage",
    "decuction": "deduction",
    "benefite": "benefit"
}

# Apply spelling corrections
print("Correcting spelling with custom corrections...")
workflow.correct_spelling(custom_corrections=custom_spelling_corrections)

# Preprocess documents
print("\nPreprocessing documents...")
workflow.preprocess_documents()

# Discover topics
print("\nDiscovering topics...")
topics_df = workflow.discover_topics(
    method="embedding_cluster",
    num_topics=7,  # Same as our number of categories for easy comparison
)
print(f"Discovered {len(topics_df['topic'].unique())} topics")

# Save the processed data with topic assignments for analysis
df_with_topics = df_sample.copy()
df_with_topics['topic'] = topics_df['topic']
df_with_topics['topic_probability'] = topics_df['topic_probability']
df_with_topics.to_csv(output_dir / "processed_insurance_data.csv", index=False)
print(f"Saved processed data to {output_dir / 'processed_insurance_data.csv'}")

# Generate various visualizations
print("\nGenerating visualizations...")

# Embeddings visualization
print("Creating embeddings visualization...")
embeddings_viz = workflow.visualize_topics(plot_type="embeddings")
embeddings_viz.write_html(str(output_dir / "insurance_embeddings.html"))
print(f"Embeddings visualization saved to {output_dir / 'insurance_embeddings.html'}")

# Topic distribution visualization
print("Creating topic distribution visualization...")
try:
    distribution_viz = workflow.visualize_topics(plot_type="distribution")
    distribution_viz.write_html(str(output_dir / "insurance_distribution.html"))
    print(f"Topic distribution visualization saved to {output_dir / 'insurance_distribution.html'}")
except Exception as e:
    print(f"Error creating topic distribution visualization: {e}")

# Time-based visualization
print("Creating time-based visualization...")
try:
    time_viz = workflow.visualize_topics(plot_type="trends")
    time_viz.write_html(str(output_dir / "insurance_trends.html"))
    print(f"Time-based visualization saved to {output_dir / 'insurance_trends.html'}")
except Exception as e:
    print(f"Error creating time-based visualization: {e}")

# Geographic visualization
print("Creating geographic visualization...")
try:
    map_viz = workflow.visualize_topics(plot_type="map")
    map_viz.write_html(str(output_dir / "insurance_map.html"))
    print(f"Geographic visualization saved to {output_dir / 'insurance_map.html'}")
except Exception as e:
    print(f"Error creating geographic visualization: {e}")

# Generate a comprehensive report
print("\nGenerating comprehensive report...")
report_path = workflow.generate_comprehensive_report(
    output_path=str(output_dir / "insurance_comprehensive_report.html"),
    title="Australian Insurance Data Analysis",
    include_interactive=True,
    include_raw_data=True,
    open_browser=False
)
print(f"Comprehensive report generated at {report_path}")

print("\nWorkflow complete! All outputs are in the 'output' directory.")
print("""
Generated files:
1. insurance_acronyms.html - Detected acronyms in the dataset
2. insurance_misspellings.html - Detected misspellings in the dataset
3. processed_insurance_data.csv - Dataset with topics assigned
4. insurance_embeddings.html - UMAP projection of document embeddings
5. insurance_distribution.html - Topic distribution visualization
6. insurance_trends.html - Time-based topic trends
7. insurance_map.html - Geographic distribution of topics
8. insurance_comprehensive_report.html - Complete analysis report
""")