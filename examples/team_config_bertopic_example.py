#!/usr/bin/env python
"""
Enhanced Team Configuration with BERTopic Integration

This example demonstrates:
1. Creating industry-specific team configurations
2. Using domain-specific acronyms and terminology
3. Integrating with BERTopic for advanced topic modeling
4. Visualizing results with industry-specific visualizations
5. Sharing configurations between teams and projects

The example uses a healthcare dataset to show how domain experts
can collaborate on terminology while data scientists focus on modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import webbrowser
from datetime import datetime, timedelta
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("meno_workflow")

# Add the parent directory to the path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Meno components
from meno.utils.team_config import (
    create_team_config,
    update_team_config,
    merge_team_configs,
    get_team_config_stats,
    export_team_acronyms
)

from meno.workflow import MenoWorkflow

# Optional BERTopic components - gracefully handle if not available
try:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not available. Some visualizations will be limited.")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
bertopic_dir = output_dir / "bertopic"
bertopic_dir.mkdir(exist_ok=True)

print("🏥 Healthcare Team Configuration with BERTopic Integration")
print("==========================================================")

# Step 1: Create specialized healthcare team configurations
print("\n📋 Creating specialized healthcare team configurations...")

# Clinical terminology team configuration
clinical_config_path = output_dir / "clinical_team_config.yaml"
clinical_config = create_team_config(
    team_name="Clinical Terminology",
    acronyms={
        # Clinical abbreviations
        "HPI": "History of Present Illness",
        "PMH": "Past Medical History",
        "FH": "Family History",
        "SH": "Social History",
        "ROS": "Review of Systems",
        "VSS": "Vital Signs Stable",
        "HTN": "Hypertension",
        "DM": "Diabetes Mellitus",
        "COPD": "Chronic Obstructive Pulmonary Disease",
        "CHF": "Congestive Heart Failure",
        "MI": "Myocardial Infarction",
        "CVA": "Cerebrovascular Accident",
        "A&O": "Alert and Oriented",
        "SOB": "Shortness of Breath",
        "CP": "Chest Pain",
        "N/V": "Nausea/Vomiting",
        "BM": "Bowel Movement",
        "PRN": "As Needed",
        "q.d.": "Once Daily",
        "b.i.d.": "Twice Daily"
    },
    spelling_corrections={
        "hyerptension": "hypertension",
        "diabeties": "diabetes",
        "patinet": "patient",
        "medicaiton": "medication",
        "symtpoms": "symptoms",
        "diaganosis": "diagnosis",
        "labratory": "laboratory",
        "proceedure": "procedure",
        "perscription": "prescription",
        "docter": "doctor"
    },
    output_path=clinical_config_path
)

# Insurance terminology team configuration
insurance_config_path = output_dir / "insurance_team_config.yaml"
insurance_config = create_team_config(
    team_name="Healthcare Insurance",
    acronyms={
        # Insurance abbreviations
        "EOB": "Explanation of Benefits",
        "OOP": "Out of Pocket",
        "CDHP": "Consumer-Driven Health Plan",
        "HDHP": "High Deductible Health Plan",
        "HSA": "Health Savings Account",
        "FSA": "Flexible Spending Account",
        "HRA": "Health Reimbursement Arrangement",
        "PPO": "Preferred Provider Organization",
        "HMO": "Health Maintenance Organization",
        "EPO": "Exclusive Provider Organization",
        "POS": "Point of Service",
        "UCR": "Usual, Customary, and Reasonable",
        "COB": "Coordination of Benefits",
        "TPA": "Third Party Administrator",
        "PBM": "Pharmacy Benefits Manager",
        "CPT": "Current Procedural Terminology",
        "ICD": "International Classification of Diseases",
        "NDC": "National Drug Code",
        "PMPM": "Per Member Per Month",
        "MLR": "Medical Loss Ratio"
    },
    spelling_corrections={
        "deductable": "deductible",
        "benifit": "benefit",
        "premeium": "premium",
        "coinsurnce": "coinsurance",
        "formulery": "formulary",
        "authroization": "authorization",
        "eligable": "eligible",
        "dependant": "dependent",
        "providor": "provider",
        "enrollement": "enrollment"
    },
    output_path=insurance_config_path
)

# Step 2: Merge configurations for comprehensive healthcare terminology
print("\n🔄 Merging team configurations for comprehensive healthcare terminology...")

merged_config_path = output_dir / "healthcare_combined_config.yaml"
merged_config = merge_team_configs(
    configs=[clinical_config_path, insurance_config_path],
    team_name="Healthcare Combined",
    output_path=merged_config_path
)

# Display statistics about the merged configuration
stats = get_team_config_stats(merged_config_path)
print(f"\nCombined Healthcare Configuration:")
print(f"- Team: {stats['team_name']}")
print(f"- Acronyms: {stats['acronym_count']}")
print(f"- Spelling corrections: {stats['spelling_correction_count']}")
print(f"- Created: {stats['created']}")

# Export the acronyms for sharing with other teams
acronyms_json_path = output_dir / "healthcare_acronyms.json"
export_team_acronyms(
    config_path=merged_config_path,
    output_format="json",
    output_path=acronyms_json_path
)
print(f"\nExported healthcare acronyms to {acronyms_json_path}")

# Step 3: Generate synthetic healthcare dataset
print("\n🧪 Generating synthetic healthcare dataset...")

# Create patient demographics
n_patients = 100
patients = []

# Patient demographics data
genders = ["Male", "Female", "Other"]
age_ranges = ["18-30", "31-45", "46-60", "61-75", "76+"]
insurance_types = ["Medicare", "Medicaid", "Private-PPO", "Private-HMO", "Private-HDHP", "Self-Pay"]
visit_types = ["Annual Exam", "Follow-up", "New Complaint", "Urgent Care", "Emergency"]
departments = ["Primary Care", "Cardiology", "Pulmonology", "Neurology", "Gastroenterology", "Endocrinology"]

# Clinical templates with placeholders
clinical_templates = [
    "Patient presents with {symptom}. PMH significant for {condition1} and {condition2}. Currently taking {medication1} and {medication2}. VSS with BP {bp} and HR {hr}. A&O x3. {assessment}.",
    "{visit_type} for {gender} patient with {condition1}. Patient reports {symptom}. {medication1} dosage adjusted. {assessment}.",
    "Follow-up for {condition1}. Patient reports {improvement} since starting {medication1}. {assessment}.",
    "New patient with c/o {symptom} for {duration}. FH positive for {condition1}. SH: {social}. {assessment}.",
    "Patient with {condition1} presents with {symptom}. ROS otherwise negative. Labs show {labs}. {assessment}.",
    "Emergency visit for {symptom}. Patient with {condition1} and {condition2}. {assessment}."
]

# Insurance templates with placeholders
insurance_templates = [
    "Claim #{claim_id} processed for {cpt_code}. Patient OOP expense: ${oop_cost}. {coverage_note}.",
    "Prior authorization for {procedure} {auth_status}. {auth_note}.",
    "Appeal for claim #{claim_id} {appeal_status}. {appeal_note}.",
    "Claim #{claim_id} for {cpt_code} processed with {adjudication_note}. Patient responsibility: ${oop_cost}.",
    "Member eligibility verified for {procedure}. Estimated patient responsibility: ${oop_cost}. {coverage_note}.",
    "Claim #{claim_id} denied due to {denial_reason}. Patient notified via EOB."
]

# Fill-in content
symptoms = ["chest pain", "shortness of breath", "abdominal pain", "headache", "dizziness", 
           "nausea", "fatigue", "joint pain", "back pain", "fever"]
conditions = ["hypertension", "diabetes", "hyperlipidemia", "COPD", "asthma", 
             "depression", "anxiety", "CHF", "atrial fibrillation", "hypothyroidism"]
medications = ["lisinopril", "metformin", "atorvastatin", "levothyroxine", "albuterol", 
              "sertraline", "metoprolol", "amlodipine", "omeprazole", "prednisone"]
assessments = ["Will continue current treatment plan", "Medication adjusted", 
              "Referred to specialist", "Ordered additional labs", "Follow-up in 3 months",
              "Recommended lifestyle changes", "Symptoms improved", "Condition stable"]
improvements = ["significant improvement", "moderate improvement", "slight improvement", "no improvement", "worsening"]
durations = ["2 days", "1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year"]
social_history = ["non-smoker", "former smoker", "current smoker", "occasional alcohol use", 
                 "denies alcohol", "sedentary lifestyle", "physically active"]
lab_results = ["elevated glucose", "elevated HbA1c", "normal CBC", "elevated WBC", 
              "normal metabolic panel", "elevated lipids", "normal thyroid function", 
              "abnormal liver enzymes"]

# Insurance-specific content
cpt_codes = ["99213", "99214", "99215", "73030", "71046", "80053", "85025", "93000"]
authorization_status = ["approved", "pending", "denied", "additional information requested"]
appeal_status = ["approved", "denied", "pending review"]
denial_reasons = ["non-covered service", "out-of-network provider", "no prior authorization", 
                 "experimental procedure", "maximum benefits exceeded", "coordination of benefits needed"]
coverage_notes = ["Patient has met deductible", "Patient has not met deductible", 
                 "Copay applied", "Coinsurance applied", "Coverage limited to 80%"]

# Generate synthetic patient notes
for i in range(n_patients):
    patient_id = f"PT{10000 + i}"
    gender = random.choice(genders)
    age_range = random.choice(age_ranges)
    insurance = random.choice(insurance_types)
    visit_date = datetime.now() - timedelta(days=random.randint(1, 365))
    department = random.choice(departments)
    visit_type = random.choice(visit_types)
    
    # Generate clinical note
    template = random.choice(clinical_templates)
    condition1 = random.choice(conditions)
    condition2 = random.choice([c for c in conditions if c != condition1])
    medication1 = random.choice(medications)
    medication2 = random.choice([m for m in medications if m != medication1])
    
    clinical_note = template.format(
        gender=gender.lower(),
        symptom=random.choice(symptoms),
        condition1=condition1,
        condition2=condition2,
        medication1=medication1,
        medication2=medication2,
        bp=f"{random.randint(110, 160)}/{random.randint(60, 100)}",
        hr=str(random.randint(60, 100)),
        assessment=random.choice(assessments),
        visit_type=visit_type,
        improvement=random.choice(improvements),
        duration=random.choice(durations),
        social=random.choice(social_history),
        labs=random.choice(lab_results)
    )
    
    # Generate insurance note
    insurance_template = random.choice(insurance_templates)
    claim_id = random.randint(100000, 999999)
    cpt_code = random.choice(cpt_codes)
    oop_cost = random.randint(0, 500)
    
    insurance_note = insurance_template.format(
        claim_id=claim_id,
        cpt_code=cpt_code,
        oop_cost=oop_cost,
        procedure=f"{random.choice(['diagnostic', 'therapeutic', 'preventive'])} {random.choice(['imaging', 'procedure', 'test', 'consultation'])}",
        auth_status=random.choice(authorization_status),
        auth_note=f"Decision based on {random.choice(['medical necessity', 'benefit coverage', 'clinical guidelines'])}",
        appeal_status=random.choice(appeal_status),
        appeal_note=f"Based on {random.choice(['clinical review', 'policy interpretation', 'additional documentation'])}",
        adjudication_note=f"Processed per {random.choice(['plan benefits', 'provider contract', 'usual and customary rates'])}",
        coverage_note=random.choice(coverage_notes),
        denial_reason=random.choice(denial_reasons)
    )
    
    # Combine notes based on department
    if department in ["Primary Care", "Cardiology", "Pulmonology", "Neurology", "Gastroenterology", "Endocrinology"]:
        # Clinical departments get primarily clinical notes with some insurance info
        combined_note = f"{clinical_note} {insurance_note if random.random() < 0.3 else ''}"
        note_type = "Clinical"
    else:
        # Insurance departments get primarily insurance notes with some clinical info
        combined_note = f"{insurance_note} {clinical_note if random.random() < 0.3 else ''}"
        note_type = "Insurance"
    
    # Add some misspellings randomly (with 20% chance)
    if random.random() < 0.2:
        # Clinical misspellings
        if "hypertension" in combined_note:
            combined_note = combined_note.replace("hypertension", "hyerptension")
        if "patient" in combined_note:
            combined_note = combined_note.replace("patient", "patinet")
        if "symptoms" in combined_note:
            combined_note = combined_note.replace("symptoms", "symtpoms")
        
        # Insurance misspellings
        if "deductible" in combined_note:
            combined_note = combined_note.replace("deductible", "deductable")
        if "benefit" in combined_note:
            combined_note = combined_note.replace("benefit", "benifit")
        if "eligible" in combined_note:
            combined_note = combined_note.replace("eligible", "eligable")
    
    # Add some acronyms that need expansion (with 50% chance)
    if random.random() < 0.5:
        acronym_phrases = [
            "Patient OOP maximum reached.",
            "EOB sent to patient's address.",
            "PMH includes HTN and DM.",
            "Patient reports SOB and CP.",
            "PMPM cost increased by 5%.",
            "Follow-up with PCP in 2 weeks.",
            "Patient enrolled in HDHP with HSA.",
            "MLR requirements met for Q4.",
            "COB with secondary insurance pending.",
            "Patient A&O x3 during visit."
        ]
        combined_note += f" {random.choice(acronym_phrases)}"
    
    # Add to patient list
    patients.append({
        "patient_id": patient_id,
        "gender": gender,
        "age_range": age_range,
        "insurance": insurance,
        "visit_date": visit_date,
        "department": department,
        "visit_type": visit_type,
        "note_type": note_type,
        "note": combined_note
    })

# Create DataFrame
healthcare_df = pd.DataFrame(patients)
print(f"Created dataset with {len(healthcare_df)} patient notes")
print(healthcare_df[["patient_id", "gender", "age_range", "department"]].head(3))

# Optional: Save dataset
csv_path = output_dir / "healthcare_notes.csv"
healthcare_df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")

# Step 4: Apply team configuration with the workflow
print("\n🔍 Applying team configuration with workflow...")

try:
    # Initialize workflow with merged team config
    workflow = MenoWorkflow(config_path=merged_config_path)

    # Load data
    workflow.load_data(
        data=healthcare_df,
        text_column="note",
        time_column="visit_date",
        category_column="department",
    )

    # Generate acronym report
    print("Generating acronym report...")
    acronym_report_path = workflow.generate_acronym_report(
        output_path=output_dir / "healthcare_acronyms.html",
        min_count=1,  # Set lower threshold for demo
        open_browser=False
    )
    print(f"Acronym report generated at {acronym_report_path}")

    # Generate spelling report
    print("Generating misspelling report...")
    spelling_report_path = workflow.generate_misspelling_report(
        output_path=output_dir / "healthcare_spelling.html",
        min_count=1,  # Set lower threshold for demo
        open_browser=False
    )
    print(f"Spelling report generated at {spelling_report_path}")

    # Apply corrections to the data
    print("Applying corrections from team configuration...")
    workflow.expand_acronyms()
    workflow.correct_spelling()

    # Preprocess documents
    print("Preprocessing documents...")
    workflow.preprocess_documents(
        additional_stopwords=[
            "patient", "note", "hospital", "medical", "health", "visit", "doctor",
            "clinical", "insurance", "please", "provider", "plan", "care"
        ]
    )

    # Step 5: BERTopic Integration (if available)
    if BERTOPIC_AVAILABLE:
        print("\n🧠 Integrating with BERTopic for advanced topic modeling...")
        
        # Get preprocessed data
        preprocessed_df = workflow.get_preprocessed_data()
        
        # Configure BERTopic with enhanced components
        print("Configuring BERTopic model...")
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
            bm25_weighting=True
        )
        keybert_model = KeyBERTInspired()
        
        # Create and fit BERTopic model
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",  # Small, fast model
            vectorizer_model=ctfidf_model,
            representation_model=keybert_model,
            nr_topics=8,  # Set lower for demo
            calculate_probabilities=True,
            verbose=True
        )
        
        print("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(
            preprocessed_df["processed_text"].tolist()
        )
        
        # Add topics back to DataFrame and update workflow
        preprocessed_df["topic"] = [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics]
        preprocessed_df["topic_probability"] = probs.max(axis=1)
        
        # Update the workflow with BERTopic results
        workflow.set_topic_assignments(preprocessed_df[["topic", "topic_probability"]])
        
        # Generate BERTopic visualizations
        print("Generating BERTopic visualizations...")
        topic_model.visualize_topics().write_html(bertopic_dir / "healthcare_topic_similarity.html")
        topic_model.visualize_hierarchy().write_html(bertopic_dir / "healthcare_topic_hierarchy.html")
        topic_model.visualize_barchart(top_n_topics=8).write_html(bertopic_dir / "healthcare_topic_barchart.html")
        
        # Generate department-specific topics
        print("Analyzing topics by department...")
        for department in healthcare_df["department"].unique():
            dept_indices = healthcare_df["department"] == department
            if sum(dept_indices) > 5:  # Only departments with enough samples
                dept_docs = preprocessed_df.loc[dept_indices, "processed_text"].tolist()
                dept_topics = topic_model.find_topics(dept_docs)
                
                print(f"Top topics for {department}:")
                for i, (topic_num, similarity) in enumerate(zip(*dept_topics)):
                    if i < 3:  # Show top 3
                        words = topic_model.get_topic(topic_num)
                        topic_words = ", ".join([word for word, _ in words[:5]])
                        print(f"  Topic {topic_num}: {topic_words} (similarity: {similarity:.2f})")
    
    # Step 6: Generate comprehensive report
    print("\n📊 Generating comprehensive healthcare report...")
    report_path = workflow.generate_comprehensive_report(
        output_path=output_dir / "healthcare_comprehensive_report.html",
        title="Healthcare Data Analysis with Team Configuration",
        include_interactive=True,
        include_raw_data=True,
        open_browser=False
    )
    print(f"Comprehensive report generated at {report_path}")
    
    # Open report in browser if desired
    print(f"\nTo view reports, open:")
    print(f"- Acronym Report: file://{os.path.abspath(acronym_report_path)}")
    print(f"- Spelling Report: file://{os.path.abspath(spelling_report_path)}")
    print(f"- Comprehensive Report: file://{os.path.abspath(report_path)}")
    if BERTOPIC_AVAILABLE:
        print(f"- BERTopic Similarity: file://{os.path.abspath(bertopic_dir / 'healthcare_topic_similarity.html')}")
        print(f"- BERTopic Hierarchy: file://{os.path.abspath(bertopic_dir / 'healthcare_topic_hierarchy.html')}")
        print(f"- BERTopic Barchart: file://{os.path.abspath(bertopic_dir / 'healthcare_topic_barchart.html')}")

except Exception as e:
    print(f"\n⚠️ Error: {str(e)}")
    print("Some functionality may be limited. Core team configuration features should still work.")

print("\n✅ Example complete!")
print("""
This demonstration shows how healthcare organizations can:

1. Create specialized team configurations for different departments
2. Share and merge domain-specific terminology across the organization
3. Apply corrections and expansions to clinical and insurance text
4. Integrate with advanced BERTopic modeling for deeper insights
5. Generate department-specific topic analysis

Using this approach, organizations can:
- Standardize terminology across teams
- Improve text processing accuracy
- Share domain knowledge efficiently
- Generate more meaningful topic models
- Provide better insights to stakeholders
""")