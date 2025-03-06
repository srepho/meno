"""
Example of using team configuration files with Meno Workflow

This example demonstrates:
1. Creating a shareable configuration file with team-specific acronyms and settings
2. Loading the configuration file into a workflow
3. Making runtime customizations while preserving team settings
4. Saving updates back to the configuration file

This allows teams to share common settings like:
- Domain-specific acronym mappings
- Standard spelling corrections
- Local model paths for offline usage
- Visualization and report format preferences
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import webbrowser

# Add the parent directory to the path so we can import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meno.utils.config import (
    load_config,
    save_config,
    merge_configs,
    WorkflowMenoConfig
)

from meno.workflow import (
    MenoWorkflow,
    load_workflow_config,
    save_workflow_config
)

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Example 1: Create a new team configuration file

print("Creating a new team configuration file...")

# Start with a default configuration
config = load_workflow_config()

# Customize for our team's needs
config.preprocessing.acronyms.custom_mappings.update({
    # Healthcare-specific acronyms
    "HMO": "Health Maintenance Organization",
    "PPO": "Preferred Provider Organization",
    "EMR": "Electronic Medical Record",
    "PHI": "Protected Health Information",
    "HIPAA": "Health Insurance Portability and Accountability Act",
    "PCP": "Primary Care Physician",
    "OOP": "Out Of Pocket",
    "EOB": "Explanation Of Benefits",
    "FSA": "Flexible Spending Account",
    "HSA": "Health Savings Account",
    
    # Insurance-specific acronyms
    "UW": "Underwriting",
    "P&C": "Property and Casualty",
    "LOB": "Line of Business",
    "NB": "New Business",
    "DOL": "Date of Loss",
    "COI": "Certificate of Insurance",
    "ROI": "Return on Investment",
    "TPL": "Third Party Liability",
    
    # Company-specific acronyms
    "PAT": "Policy Administration Team",
    "CMT": "Claims Management Team",
    "DIT": "Data Intelligence Team",
    "QAT": "Quality Assurance Team",
    "CET": "Customer Experience Team",
    "RMT": "Risk Management Team",
})

# Add team-specific spelling corrections
config.preprocessing.spelling.custom_dictionary.update({
    "policyhlder": "policyholder",
    "recieved": "received",
    "premiume": "premium",
    "vehical": "vehicle",
    "medicaln": "medical",
    "insurence": "insurance",
    "deductible": "deductible",
    "claimform": "claim form",
    "adjstar": "adjuster",
    "benefitt": "benefit",
})

# Configure report paths and features
config.workflow.report_paths.acronym_report = str(output_dir / "health_insurance_acronyms.html")
config.workflow.report_paths.spelling_report = str(output_dir / "health_insurance_spelling.html")
config.workflow.report_paths.comprehensive_report = str(output_dir / "health_insurance_report.html")

# Set default visualization preferences
config.visualization.defaults.plot_type = "distribution"
config.visualization.time.resample_freq = "W"  # Weekly aggregation
config.visualization.category.max_categories = 10

# Update modeling defaults
config.modeling.default_num_topics = 12
config.modeling.embeddings.model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Save the team configuration to a file
team_config_path = output_dir / "health_insurance_team_config.yaml"
save_workflow_config(config, team_config_path)
print(f"Team configuration saved to {team_config_path}")

# Example 2: Use the team configuration in a workflow

print("\nUsing the team configuration in a workflow...")

# Create sample health insurance data
data = pd.DataFrame({
    "text": [
        "Patient reached OOP maximum after hospital stay according to EOB.",
        "Policyhlder recieved notice about premiume increase for vehical insurance.",
        "The UW department flagged this application due to pre-existing conditions.",
        "Our team processed 50 claims this week, with 10 requiring manual review by the CMT.",
        "The PAT and DIT are collaborating on a new policy verification system.",
        "According to HIPAA regulations, we cannot disclose PHI without authorization.",
        "Claims with high deductible health plans require special handling by the adjstar.",
        "The claimform for medicaln benefits must be submitted within 30 days.",
        "Our HSA product has seen increased adoption this quarter.",
        "The PCP referral is needed before seeing a specialist under this HMO plan.",
        "TPL investigation is ongoing for the auto accident claim."
    ],
    "date": pd.date_range(start="2023-01-01", periods=11, freq="W"),
    "department": ["Claims", "Billing", "Underwriting", "Operations", "IT", 
                  "Compliance", "Claims", "Benefits", "Products", "Provider Networks", "Legal"],
    "region": ["Northeast", "South", "Midwest", "West", "Northeast", "South", 
              "Midwest", "West", "Northeast", "South", "Midwest"]
})

# Initialize workflow with team config
try:
    workflow = MenoWorkflow(config_path=team_config_path)

    # Load the data
    workflow.load_data(
        data=data,
        text_column="text",
        time_column="date",
        category_column="department",
        geo_column="region"
    )

    # Generate acronym report
    print("Generating acronym report...")
    acronym_report_path = workflow.generate_acronym_report()
    print(f"Acronym report generated at {acronym_report_path}")

    # Open the report in browser
    webbrowser.open(f'file://{os.path.abspath(acronym_report_path)}')

    # Generate spelling report
    print("Generating spelling report...")
    spelling_report_path = workflow.generate_misspelling_report()
    print(f"Spelling report generated at {spelling_report_path}")

    # Open the report in browser
    webbrowser.open(f'file://{os.path.abspath(spelling_report_path)}')

    # Expand acronyms and correct spelling
    print("Applying corrections to the data...")
    workflow.expand_acronyms()
    workflow.correct_spelling()

    # Show sample of processed text
    print("\nSample of processed text:")
    processed_sample = workflow.documents["text"].iloc[0:3]
    for i, text in enumerate(processed_sample):
        print(f"[{i+1}] {text}")

    # Try to process text and discover topics (may not work due to dependencies)
    try:
        print("\nDiscovering topics...")
        workflow.preprocess_documents()
        topics_df = workflow.discover_topics()
        print(f"Discovered {len(topics_df['topic'].unique())} topics")

        # Generate report with time trends visualization
        print("Generating comprehensive report...")
        report_path = workflow.generate_comprehensive_report()
        print(f"Report generated at {report_path}")
    except Exception as e:
        print(f"\nUnable to complete topic modeling due to: {str(e)}")
        print("This is expected if full dependencies aren't installed.")
        print("The core workflow features (acronym detection and spelling correction) are working!")

except Exception as e:
    print(f"\nWorkflow initialization failed: {str(e)}")
    print("Falling back to configuration-only demonstration...")

# Example 3: Update existing configuration with new domain terminology

print("\nUpdating team configuration with new industry terms...")

# Load existing config
config = load_workflow_config(config_path=team_config_path)

# Add new acronyms discovered during analysis
config.preprocessing.acronyms.custom_mappings.update({
    "MLR": "Medical Loss Ratio",
    "PMPM": "Per Member Per Month",
    "COB": "Coordination of Benefits",
    "SNF": "Skilled Nursing Facility",
    "UM": "Utilization Management",
    "PA": "Prior Authorization"
})

# Add new spelling corrections discovered in documents
config.preprocessing.spelling.custom_dictionary.update({
    "eligable": "eligible",
    "coinsurnce": "coinsurance",
    "reimbursmnt": "reimbursement",
    "dependant": "dependent",
    "specalist": "specialist"
})

# Save the updated configuration
save_workflow_config(config, team_config_path)
print(f"Updated configuration saved to {team_config_path}")

print("\nExample complete!")
print("""
This demonstration shows how healthcare insurance teams can:
1. Create domain-specific terminology configurations
2. Share these across the organization via version control
3. Identify acronyms and misspellings in their documents
4. Iteratively improve their domain dictionary
5. Generate consistent, corrected documents for analysis

Using this approach, organizations build valuable dictionaries of 
domain-specific terms that improve text processing and analysis.
""")