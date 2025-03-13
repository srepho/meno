# Time Series Analysis - Visualizing Topic Trends Over Time
#
# This example demonstrates how to visualize topic trends over time
# and create animated visualizations for temporal topic data.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meno import MenoTopicModeler
from meno.visualization.time_series import (
    plot_topic_trends,
    plot_topic_evolution,
    plot_topic_composition_over_time
)

# Create sample temporal data
np.random.seed(42)
num_docs = 1000
topics = ["Technology", "Healthcare", "Finance", "Education", "Entertainment"]

# Generate random dates spanning 2 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
days_range = (end_date - start_date).days
random_days = np.random.randint(0, days_range, num_docs)
dates = [start_date + timedelta(days=day) for day in random_days]
dates.sort()  # Sort chronologically

# Generate random topic distributions that evolve over time
# Technology starts strong, then decreases
# Healthcare increases over time
# Finance remains stable
# Education has seasonal pattern
# Entertainment grows in the second year

def get_topic_weight(topic, date_value):
    progress = (date_value - start_date).days / days_range
    
    if topic == "Technology":
        return 0.4 - (0.2 * progress) + np.random.normal(0, 0.05)
    elif topic == "Healthcare":
        return 0.1 + (0.3 * progress) + np.random.normal(0, 0.05)
    elif topic == "Finance":
        return 0.2 + np.random.normal(0, 0.05)
    elif topic == "Education":
        # Seasonal pattern (higher in fall and spring)
        month = date_value.month
        season_effect = 0.1 if month in [3, 4, 5, 9, 10, 11] else 0
        return 0.15 + season_effect + np.random.normal(0, 0.05)
    elif topic == "Entertainment":
        # Growth in second year
        second_year = 1 if date_value > datetime(2023, 1, 1) else 0
        return 0.1 + (0.15 * second_year) + np.random.normal(0, 0.05)

# Generate documents and their topic distributions
documents = []
document_topics = []
topic_weights = []
timestamps = []

for i in range(num_docs):
    date = dates[i]
    weights = [max(0, get_topic_weight(topic, date)) for topic in topics]
    # Normalize weights to sum to 1
    weights = [w/sum(weights) for w in weights]
    
    # Assign dominant topic
    dominant_topic = topics[np.argmax(weights)]
    
    # Generate document text (simplified for example)
    doc = f"This is document {i} about {dominant_topic} created on {date.strftime('%Y-%m-%d')}"
    documents.append(doc)
    document_topics.append(dominant_topic)
    topic_weights.append(weights)
    timestamps.append(date)

# Create DataFrame
df = pd.DataFrame({
    "document": documents,
    "topic": document_topics,
    "date": timestamps
})

# Add topic probability columns
for i, topic in enumerate(topics):
    df[f"{topic}_prob"] = [weights[i] for weights in topic_weights]

print(f"Created sample dataset with {len(df)} documents across {len(topics)} topics")
print(df.head())

# Initialize modeler with the pre-assigned topics
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="document")

# Since we already have topics assigned, we'll manually set document topics
topic_df = df[["document", "topic"]].copy()
topic_df.columns = ["Document", "Topic"]
modeler.topic_assignments = topic_df

# Create time series visualization
print("\nCreating time series visualizations...")

# Daily topic counts (sampled to weekly for smoother trends)
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
weekly_counts = df.groupby(['year', 'week', 'topic']).size().reset_index(name='count')

# Convert back to datetime for easier plotting
weekly_counts['date'] = weekly_counts.apply(
    lambda row: datetime.strptime(f"{int(row['year'])}-W{int(row['week'])}-1", "%Y-W%W-%w"), 
    axis=1
)

# Create topic trends line chart
fig_trends = plot_topic_trends(
    weekly_counts,
    date_column='date',
    topic_column='topic',
    value_column='count',
    title="Topic Trends Over Time (Weekly)"
)
fig_trends.write_html("topic_trends_over_time.html")
print("Created topic trends visualization (saved to topic_trends_over_time.html)")

# Create topic composition area chart
topic_columns = [f"{topic}_prob" for topic in topics]
composition_data = df.groupby(pd.Grouper(key='date', freq='MS'))[topic_columns].mean().reset_index()
composition_data.columns = ['date'] + topics  # Rename for easier plotting

fig_comp = plot_topic_composition_over_time(
    composition_data,
    date_column='date',
    topic_columns=topics,
    title="Topic Composition Over Time (Monthly)"
)
fig_comp.write_html("topic_composition_over_time.html")
print("Created topic composition visualization (saved to topic_composition_over_time.html)")

# Create topic evolution animation (showing how topic distributions change)
# For this, we'll use a more condensed time frame (quarterly)
df['quarter'] = df['date'].dt.to_period('Q')
quarterly_data = df.groupby('quarter')[topic_columns].mean().reset_index()
quarterly_data['date'] = quarterly_data['quarter'].apply(lambda x: x.start_time)

try:
    fig_evolution = plot_topic_evolution(
        quarterly_data,
        date_column='date',
        topic_columns=topics,
        title="Topic Evolution (Quarterly)"
    )
    fig_evolution.write_html("topic_evolution_animated.html")
    print("Created topic evolution animation (saved to topic_evolution_animated.html)")
except Exception as e:
    print(f"Could not create topic evolution animation: {e}")

print("\nAll time series visualizations complete!")
print("Open the HTML files in a browser to explore the interactive visualizations.")