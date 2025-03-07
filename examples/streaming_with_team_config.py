#!/usr/bin/env python
"""
Memory-Efficient Processing with Team Configuration for Large Datasets

This example demonstrates:
1. Using team configurations for domain-specific terminology
2. Processing large datasets with streaming capabilities
3. Memory-efficient embedding generation with quantization
4. Combining batched processing with topic modeling
5. Advanced visualizations for large-scale data

This approach is ideal for:
- Very large datasets (millions of documents)
- Low-memory environments (laptops, small cloud instances)
- Collaborative teams with shared domain knowledge
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import time
import logging
import tempfile
from datetime import datetime, timedelta
import random
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("meno_streaming")

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Meno components
from meno.utils.team_config import create_team_config
from meno.modeling.streaming_processor import StreamingProcessor
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler

# Create output directories
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

temp_dir = Path("output/temp")
temp_dir.mkdir(exist_ok=True)

print("📊 Memory-Efficient Processing with Team Configuration")
print("=====================================================")

# Step 1: Create industry-specific team configuration
print("\n📋 Creating financial services team configuration...")

finance_config_path = output_dir / "finance_team_config.yaml"
finance_config = create_team_config(
    team_name="Financial Services",
    acronyms={
        # Banking acronyms
        "KYC": "Know Your Customer",
        "AML": "Anti-Money Laundering",
        "CTF": "Counter-Terrorism Financing",
        "FATCA": "Foreign Account Tax Compliance Act",
        "CDD": "Customer Due Diligence",
        "EDD": "Enhanced Due Diligence",
        "PEP": "Politically Exposed Person",
        "STP": "Straight-Through Processing",
        "ACH": "Automated Clearing House",
        "SWIFT": "Society for Worldwide Interbank Financial Telecommunication",
        
        # Investment acronyms
        "IPO": "Initial Public Offering",
        "M&A": "Mergers and Acquisitions",
        "PE": "Private Equity",
        "VC": "Venture Capital",
        "AUM": "Assets Under Management",
        "NAV": "Net Asset Value",
        "ROI": "Return on Investment",
        "YTD": "Year to Date",
        "MoM": "Month over Month",
        "YoY": "Year over Year",
        
        # Compliance acronyms
        "KPI": "Key Performance Indicator",
        "SLA": "Service Level Agreement",
        "CRM": "Customer Relationship Management",
        "ETF": "Exchange-Traded Fund",
        "SOX": "Sarbanes-Oxley Act",
        "GDPR": "General Data Protection Regulation",
    },
    spelling_corrections={
        "invstment": "investment",
        "portfoio": "portfolio",
        "finanical": "financial",
        "clinet": "client",
        "acuisition": "acquisition",
        "divdend": "dividend",
        "exchagne": "exchange",
        "marekt": "market",
        "securites": "securities",
        "mortage": "mortgage",
        "tranaction": "transaction",
        "statment": "statement",
        "witdrawal": "withdrawal",
        "ballance": "balance",
        "depositt": "deposit",
    },
    output_path=finance_config_path
)

print(f"Team configuration created at {finance_config_path}")

# Step 2: Generate a large synthetic financial services dataset
print("\n🧪 Generating large synthetic financial dataset...")

# Helper functions to generate large dataset
def generate_financial_text(size="medium"):
    """Generate synthetic financial text of different sizes."""
    
    # Financial terms and phrases
    financial_terms = [
        "investment portfolio", "asset allocation", "market volatility",
        "risk management", "financial planning", "retirement savings",
        "estate planning", "tax optimization", "wealth management",
        "dividend income", "equity markets", "fixed income", "bond yields",
        "interest rates", "credit rating", "mutual funds", "hedge funds",
        "capital gains", "market capitalization", "liquidity",
    ]
    
    # Financial activities
    activities = [
        "purchased shares", "sold securities", "rebalanced portfolio",
        "transferred funds", "executed trade", "allocated assets",
        "approved loan", "processed transaction", "reviewed performance",
        "updated forecast", "conducted analysis", "prepared report",
        "evaluated risk", "assessed compliance", "documented findings"
    ]
    
    # Financial institutions
    institutions = [
        "Goldman Sachs", "JPMorgan Chase", "Morgan Stanley", "Wells Fargo",
        "Bank of America", "Citigroup", "BlackRock", "Vanguard Group",
        "State Street", "Fidelity Investments", "Charles Schwab"
    ]
    
    # Common acronyms (some misspelled)
    acronyms = [
        "ROI", "IPO", "M&A", "PE", "VC", "AUM", "NAV", "YTD", "MoM", "YoY",
        "KYC", "AML", "CTF", "KPI", "SLA", "CRM", "ETF"
    ]
    
    # Generate text based on size
    if size == "small":
        # Generate a short comment (30-50 words)
        num_sentences = random.randint(1, 2)
        text_parts = []
        
        for _ in range(num_sentences):
            institution = random.choice(institutions)
            activity = random.choice(activities)
            term = random.choice(financial_terms)
            acronym = random.choice(acronyms) if random.random() < 0.3 else ""
            
            if acronym:
                text_parts.append(f"{institution} {activity} related to {term}. {acronym} metrics were updated accordingly.")
            else:
                text_parts.append(f"{institution} {activity} related to {term}.")
        
        return " ".join(text_parts)
    
    elif size == "medium":
        # Generate medium text (100-150 words)
        num_paragraphs = 1
        num_sentences = random.randint(4, 6)
        
    else:  # large
        # Generate large text (300-500 words)
        num_paragraphs = random.randint(2, 3)
        num_sentences = random.randint(6, 10)
    
    # Generate paragraphs
    paragraphs = []
    for _ in range(num_paragraphs):
        sentences = []
        for _ in range(num_sentences):
            institution = random.choice(institutions)
            activity = random.choice(activities)
            term1 = random.choice(financial_terms)
            term2 = random.choice(financial_terms)
            acronym = random.choice(acronyms) if random.random() < 0.4 else ""
            
            sentence_templates = [
                f"{institution} {activity} related to {term1} and {term2}.",
                f"The {term1} strategy at {institution} involved {activity}.",
                f"After analyzing {term1}, {institution} decided to {activity}.",
                f"Based on {term1} performance, {activity} was recommended by {institution}.",
                f"{activity} was completed for {term1} in accordance with {institution} policies."
            ]
            
            sentence = random.choice(sentence_templates)
            if acronym and random.random() < 0.5:
                sentence += f" {acronym} metrics were evaluated as part of this process."
            
            sentences.append(sentence)
        
        paragraphs.append(" ".join(sentences))
    
    # Join paragraphs and return
    return "\n\n".join(paragraphs)

def add_misspellings(text, misspelling_rate=0.05):
    """Add random misspellings to text based on common financial misspellings."""
    if random.random() > misspelling_rate * 10:  # Only apply to 10% of documents
        return text
    
    misspellings = {
        "investment": "invstment",
        "portfolio": "portfoio",
        "financial": "finanical",
        "client": "clinet",
        "acquisition": "acuisition",
        "dividend": "divdend",
        "exchange": "exchagne",
        "market": "marekt",
        "securities": "securites",
        "mortgage": "mortage",
        "transaction": "tranaction",
        "statement": "statment",
        "withdrawal": "witdrawal",
        "balance": "ballance",
        "deposit": "depositt",
    }
    
    result_text = text
    for correct, misspelled in misspellings.items():
        if correct in result_text and random.random() < misspelling_rate:
            # Replace only some occurrences to make it more realistic
            if random.random() < 0.7:  # 70% chance to replace only the first occurrence
                result_text = result_text.replace(correct, misspelled, 1)
            else:
                result_text = result_text.replace(correct, misspelled)
    
    return result_text

# Generate synthetic data
n_documents = 10000  # Simulate a large dataset
document_sizes = ["small", "medium", "large"]
size_weights = [0.6, 0.3, 0.1]  # 60% small, 30% medium, 10% large
document_types = ["Report", "Note", "Email", "Meeting Minutes", "Analysis", "Review"]
departments = ["Wealth Management", "Investment Banking", "Commercial Banking", 
              "Asset Management", "Risk Management", "Compliance", "Research"]
regions = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"]

# Create documents in batches to simulate streaming
batch_size = 1000
all_batches = []

print(f"Generating {n_documents} financial documents in batches of {batch_size}...")

for batch_start in range(0, n_documents, batch_size):
    batch_end = min(batch_start + batch_size, n_documents)
    batch_size_actual = batch_end - batch_start
    
    batch_data = []
    for i in range(batch_size_actual):
        doc_id = f"DOC-{batch_start + i + 1:06d}"
        
        # Choose document size according to weights
        doc_size = random.choices(document_sizes, weights=size_weights)[0]
        
        # Generate base text
        base_text = generate_financial_text(size=doc_size)
        
        # Add random misspellings
        text = add_misspellings(base_text, misspelling_rate=0.05)
        
        # Generate metadata
        doc_type = random.choice(document_types)
        department = random.choice(departments)
        region = random.choice(regions)
        
        # Random date in the last 2 years
        days_ago = random.randint(1, 730)  # 2 years
        doc_date = datetime.now() - timedelta(days=days_ago)
        
        batch_data.append({
            "doc_id": doc_id,
            "text": text,
            "type": doc_type,
            "department": department,
            "region": region,
            "date": doc_date,
            "size": doc_size
        })
    
    # Create batch DataFrame
    batch_df = pd.DataFrame(batch_data)
    all_batches.append(batch_df)
    
    # Report progress
    print(f"Generated batch {len(all_batches)}/{n_documents//batch_size + (1 if n_documents % batch_size else 0)}")

# Step 3: Process Using Streaming Processor
print("\n🔄 Processing documents with streaming processor...")

# Create StreamingProcessor with memory-efficient settings
processor = StreamingProcessor(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Small, fast model
    topic_model="bertopic",
    batch_size=500,  # Process in smaller batches for memory efficiency
    temp_dir=temp_dir,
    use_quantization=True,  # Use quantization for reduced memory usage
    verbose=True
)

# Process each batch to demonstrate streaming
print("\n📊 Processing data in batches...")

# Save the first batch to a CSV file for demonstration
csv_path = temp_dir / "financial_data_sample.csv"
all_batches[0].to_csv(csv_path, index=False)

# Process the CSV file using the streaming processor
print("\n🔍 Sample workflow: Processing financial_data_sample.csv...")
results = processor.process_file(
    file_path=csv_path,
    text_column="text",
    id_column="doc_id", 
    n_topics=8,  # Discover 8 topics 
    min_topic_size=10,
    topic_method="bertopic",
    save_results=True,
    output_dir=output_dir
)

print(f"\n✅ Processing complete!")
print(f"Processed {results['document_count']} documents in {results['batch_count']} batches")
print(f"Discovered {results['topic_count']} topics")
print(f"Embedding time: {results['timing']['embedding']:.2f} seconds")
print(f"Modeling time: {results['timing']['modeling']:.2f} seconds")
print(f"Total time: {results['total_time']:.2f} seconds")

# Get model information
if processor.is_fitted:
    # Get topic information
    topic_info = processor.topic_model.get_topic_info()
    
    print("\n📌 Top Topics Discovered:")
    for i, row in topic_info[topic_info["Topic"] >= 0].head(5).iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        name = row["Name"]
        print(f"Topic {topic_id}: {name} ({count} documents)")
    
    # Step 4: Demonstrate how to apply team configuration to clean text
    print("\n🧹 Applying team configuration for text cleaning...")
    
    # Load the first few documents from the first batch
    sample_docs = all_batches[0]["text"].head(3).tolist()
    sample_ids = all_batches[0]["doc_id"].head(3).tolist()
    
    # Show original text with potential issues
    print("\nOriginal sample documents:")
    for i, (doc_id, text) in enumerate(zip(sample_ids, sample_docs)):
        print(f"\n[{doc_id}] {text[:150]}...")
    
    # Load finance team configuration acronyms and corrections
    import yaml
    with open(finance_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    acronyms = config["preprocessing"]["acronyms"]["custom_mappings"]
    corrections = config["preprocessing"]["spelling"]["custom_dictionary"]
    
    # Apply corrections manually (in a real scenario, this would be done with MenoWorkflow)
    corrected_docs = []
    for doc in sample_docs:
        corrected = doc
        
        # Apply spelling corrections
        for misspelled, correct in corrections.items():
            corrected = corrected.replace(misspelled, correct)
        
        # Expand acronyms (simple approach for demonstration)
        for acronym, expansion in acronyms.items():
            # Only replace standalone acronyms (with word boundaries)
            import re
            corrected = re.sub(r'\b' + re.escape(acronym) + r'\b', 
                              f"{acronym} ({expansion})", corrected)
        
        corrected_docs.append(corrected)
    
    # Show corrected texts
    print("\nCorrected sample documents:")
    for i, (doc_id, text) in enumerate(zip(sample_ids, corrected_docs)):
        print(f"\n[{doc_id}] {text[:200]}...")
    
    # Step 5: Create interactive visualization with team-specific topics
    try:
        from bertopic import BERTopic
        from meno.visualization.bertopic_viz import (
            create_bertopic_topic_distribution,
            create_bertopic_categories_comparison
        )
        
        print("\n📊 Creating visualizations with team-specific topics...")
        
        # Get sample for visualization
        sample_df = all_batches[0]
        
        # Assign topics (use transform since the model is already fitted)
        docs = sample_df["text"].tolist()
        embeddings = processor.embedding_model.embed_documents(docs)
        topics, probs = processor.topic_model.transform(docs, embeddings=embeddings)
        
        # Add topics to DataFrame
        sample_df["topic"] = topics
        
        # Create topic distribution visualization
        dist_fig = create_bertopic_topic_distribution(
            model=processor.topic_model.model,
            topics=topics,
            title="Financial Document Topic Distribution",
            width=800,
            height=500
        )
        
        if dist_fig:
            dist_path = output_dir / "financial_topic_distribution.html"
            dist_fig.write_html(str(dist_path))
            print(f"Topic distribution visualization saved to {dist_path}")
        
        # Create category comparison visualization
        cat_fig = create_bertopic_categories_comparison(
            model=processor.topic_model.model,
            docs_df=sample_df,
            topic_col="topic",
            category_col="department",
            title="Topic Distribution by Department",
            width=900,
            height=600
        )
        
        if cat_fig:
            cat_path = output_dir / "financial_department_topics.html"
            cat_fig.write_html(str(cat_path))
            print(f"Department topic visualization saved to {cat_path}")
        
    except ImportError as e:
        print(f"\n⚠️ Visualization requires additional dependencies: {e}")
        print("Install with: pip install bertopic plotly")

# Step 6: Clean up temporary files
print("\n🧹 Cleaning up temporary files...")
processor.clean_temp_files()

print("\n✅ Memory-efficient processing complete!")
print("""
This demonstration shows how to:
1. Create and use domain-specific team configurations
2. Process large financial datasets in a memory-efficient way
3. Use quantization to reduce memory requirements
4. Apply streaming processing for datasets that don't fit in memory
5. Generate interactive visualizations of the results
6. Clean up temporary files after processing

Key memory optimization techniques:
- Quantized embeddings (reduced precision)
- Batch processing (small chunks)
- Streaming from files (avoid loading everything at once)
- Temporary file cleanup (manage disk space)
- Small embedding models (faster, less memory)
""")

print(f"\nTo view visualizations, open:")
print(f"file://{os.path.abspath(output_dir / 'financial_topic_distribution.html')}")
print(f"file://{os.path.abspath(output_dir / 'financial_department_topics.html')}")