"""
A simpler test of the Meno workflow with a small synthetic dataset
to avoid dependency issues with the full Hugging Face test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import random
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We need to bypass the imports in __init__.py that might cause issues
# So import workflow directly
sys.path.append(str(Path(__file__).resolve().parent.parent / "meno"))
from workflow import MenoWorkflow

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Create synthetic data
print("Generating synthetic data...")

# Create some sample insurance claims text
claims_text = [
    "Customer's vehicle was damaged in a parking lot by a shopping cart. The policy covers this type of incident.",
    "Claimant's home flooded due to heavy rain. Water damage to first floor requiring extensive repairs.",
    "Vehicle collided with another car at an intersection. Front-end damage and possible injury to driver.",
    "Tree fell on roof during storm causing damage to shingles and gutters. No interior water damage reported.",
    "Insured slipped on ice in parking lot and broke wrist requiring treatment at hospital.",
    "Kitchen fire damaged cabinets and appliances. Smoke damage throughout home requiring professional cleaning.",
    "Policyholder's laptop was stolen from car while parked at shopping mall. Filed police report.",
    "Hail storm damaged roof and vehicle. Multiple claims being filed for same incident.",
    "Customer reports water leak from upstairs bathroom causing ceiling damage in living room below.",
    "Jewelry was stolen during home burglary. Items include engagement ring (insured) and watch.",
    "Rear-end collision at stop light. Customer not at fault. Other driver cited by police.",
    "Wind storm blew down fence on property line. Neighbors disputing responsibility for replacement.",
    "Theft of bicycle from garage. Door was left unlocked overnight according to insured.",
    "Customer backed into pole in parking garage. Rear bumper and taillight damage on vehicle.",
    "Dog bit visitor to home requiring stitches. Medical payments coverage being requested.",
    "Hail damage to roof of insured property. Multiple similar claims in neighborhood after storm."
]

# Create some synthetic insurance acronyms in some of the texts
more_claims = [
    "The CTP insurance should cover the TPD claim according to our PDS.",
    "The ROI on this premium has been declining according to our KPI metrics.",
    "Our CEO instructed the CFO to review all P&C claims this quarter.",
    "The insured requested an EOB for their medical claim.",
    "The CSR processed the NCD for the customer's auto policy renewal.",
    "Please check the MOB and YOB figures for this LOB segment.",
    "The IT department is upgrading our CRM system for better customer management.",
    "The UW department rejected the application due to insuficient data.",
    "Our EBITDA has improved despite higher CAC in our insurance vertical."
]

# Create some synthetic misspellings in some of the texts
misspelled_claims = [
    "The policyhlder recieved a premiume notice for their vehical insurance.",
    "The custmer filled a cliam for water damaage to their property.",
    "The agent forgott to proccess the covrage extension request.",
    "The adjstar calculated the decuction incorrectly on the claimform.",
    "The policey includes protection for personall property and liabiltiy."
]

# Combine all the texts
all_texts = claims_text + more_claims + misspelled_claims
random.shuffle(all_texts)

# Create a dataframe with the texts
df = pd.DataFrame({"text": all_texts})

# Add synthetic time data - random dates within the last year
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

df['date'] = [random_date(start_date, end_date) for _ in range(len(df))]

# Add synthetic geographic data - states
states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
df['state'] = np.random.choice(states, size=len(df))

# Add synthetic category data
categories = ['Auto', 'Home', 'Health', 'Life', 'Business']
df['category'] = np.random.choice(categories, size=len(df))

print(f"Created synthetic dataset with {len(df)} records")

# Initialize the workflow
print("\nInitializing the MenoWorkflow...")
workflow = MenoWorkflow()

# Load the data
print("Loading data into workflow...")
workflow.load_data(
    data=df,
    text_column="text",
    time_column="date",
    geo_column="state",
    category_column="category"
)

# Generate acronym report
print("\nGenerating acronym report...")
acronym_report_path = workflow.generate_acronym_report(
    min_length=2,
    min_count=1,
    output_path=str(output_dir / "simple_acronyms.html"),
    open_browser=False
)
print(f"Acronym report generated at {acronym_report_path}")

# Define custom acronym mappings
custom_acronym_mappings = {
    "CTP": "Compulsory Third Party",
    "TPD": "Total and Permanent Disability",
    "PDS": "Product Disclosure Statement",
    "ROI": "Return on Investment",
    "KPI": "Key Performance Indicator",
    "CEO": "Chief Executive Officer",
    "CFO": "Chief Financial Officer",
    "P&C": "Property and Casualty",
    "EOB": "Explanation of Benefits",
    "CSR": "Customer Service Representative",
    "NCD": "No Claims Discount",
    "MOB": "Month of Business",
    "YOB": "Year of Business",
    "LOB": "Line of Business",
    "IT": "Information Technology",
    "CRM": "Customer Relationship Management",
    "UW": "Underwriting",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
    "CAC": "Customer Acquisition Cost"
}

# Expand acronyms
print("Expanding acronyms with custom mappings...")
workflow.expand_acronyms(custom_mappings=custom_acronym_mappings)

# Generate spelling report
print("\nGenerating misspelling report...")
misspelling_report_path = workflow.generate_misspelling_report(
    min_length=3,
    min_count=1,
    output_path=str(output_dir / "simple_misspellings.html"),
    open_browser=False
)
print(f"Misspelling report generated at {misspelling_report_path}")

# Define custom spelling corrections
custom_spelling_corrections = {
    "policyhlder": "policyholder",
    "recieved": "received",
    "premiume": "premium",
    "vehical": "vehicle",
    "custmer": "customer",
    "cliam": "claim",
    "damaage": "damage",
    "forgott": "forgot",
    "proccess": "process",
    "covrage": "coverage",
    "adjstar": "adjuster",
    "decuction": "deduction",
    "claimform": "claim form",
    "policey": "policy",
    "personall": "personal",
    "liabiltiy": "liability",
    "insuficient": "insufficient"
}

# Correct spelling
print("Correcting spelling with custom corrections...")
workflow.correct_spelling(custom_corrections=custom_spelling_corrections)

# Preprocess documents
print("\nPreprocessing documents...")
workflow.preprocess_documents()

# Discover topics
print("\nDiscovering topics...")
try:
    topics_df = workflow.discover_topics(
        method="embedding_cluster",
        num_topics=5,  # Same as our number of categories
    )
    print(f"Discovered {len(topics_df['topic'].unique())} topics")

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = workflow.generate_comprehensive_report(
        output_path=str(output_dir / "simple_report.html"),
        title="Simple Insurance Data Analysis",
        include_interactive=True,
        include_raw_data=True,
        open_browser=False
    )
    print(f"Comprehensive report generated at {report_path}")

except Exception as e:
    print(f"Error during topic discovery or report generation: {e}")

print("\nWorkflow test complete!")