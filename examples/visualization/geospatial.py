# Geospatial Visualization - Mapping Topics with Geographic Data
#
# This example demonstrates how to create maps and geographic visualizations
# for topic modeling results with location data.

import pandas as pd
import numpy as np
import json
from meno import MenoTopicModeler
from meno.visualization.geospatial import (
    plot_topic_map,
    plot_topic_choropleth,
    plot_topic_density_map,
    plot_topic_spacetime_heatmap
)

# Create a sample dataset with Australian postcodes
# This example uses a simplified approach with random data
# In a real application, you would load actual data with geocoded information

# Sample Australian cities and their coordinates
cities = {
    "Sydney": {"lat": -33.8688, "lon": 151.2093, "state": "NSW"},
    "Melbourne": {"lat": -37.8136, "lon": 144.9631, "state": "VIC"},
    "Brisbane": {"lat": -27.4698, "lon": 153.0251, "state": "QLD"},
    "Perth": {"lat": -31.9505, "lon": 115.8605, "state": "WA"},
    "Adelaide": {"lat": -34.9285, "lon": 138.6007, "state": "SA"},
    "Gold Coast": {"lat": -28.0167, "lon": 153.4000, "state": "QLD"},
    "Newcastle": {"lat": -32.9283, "lon": 151.7817, "state": "NSW"},
    "Canberra": {"lat": -35.2809, "lon": 149.1300, "state": "ACT"},
    "Wollongong": {"lat": -34.4331, "lon": 150.8831, "state": "NSW"},
    "Hobart": {"lat": -42.8821, "lon": 147.3272, "state": "TAS"},
    "Townsville": {"lat": -19.2590, "lon": 146.8169, "state": "QLD"},
    "Cairns": {"lat": -16.9186, "lon": 145.7781, "state": "QLD"},
    "Darwin": {"lat": -12.4634, "lon": 130.8456, "state": "NT"},
    "Toowoomba": {"lat": -27.5598, "lon": 151.9507, "state": "QLD"},
    "Ballarat": {"lat": -37.5622, "lon": 143.8503, "state": "VIC"}
}

# Sample topics related to insurance claims
topics = [
    "Vehicle Accident", 
    "Home Damage", 
    "Medical Claim", 
    "Property Theft", 
    "Natural Disaster"
]

# Topic prevalence by state (simplified for example)
topic_state_prevalence = {
    "NSW": [0.3, 0.2, 0.15, 0.25, 0.1],  # Vehicle accidents higher in NSW
    "VIC": [0.25, 0.3, 0.2, 0.15, 0.1],  # Home damage higher in VIC
    "QLD": [0.2, 0.15, 0.1, 0.15, 0.4],  # Natural disasters higher in QLD
    "WA": [0.25, 0.2, 0.2, 0.25, 0.1],
    "SA": [0.2, 0.3, 0.2, 0.2, 0.1],
    "TAS": [0.15, 0.3, 0.25, 0.2, 0.1],
    "NT": [0.2, 0.1, 0.3, 0.3, 0.1],
    "ACT": [0.3, 0.25, 0.2, 0.2, 0.05]
}

# Create sample data
np.random.seed(42)
num_docs = 1000

documents = []
topic_list = []
city_list = []
state_list = []
lat_list = []
lon_list = []
postcode_list = []
dates = []

from datetime import datetime, timedelta
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
days_range = (end_date - start_date).days

for i in range(num_docs):
    # Select random city
    city = np.random.choice(list(cities.keys()))
    city_data = cities[city]
    state = city_data["state"]
    
    # Add some noise to coordinates (to spread points)
    lat = city_data["lat"] + np.random.normal(0, 0.03)
    lon = city_data["lon"] + np.random.normal(0, 0.03)
    
    # Generate random postcode (simplified)
    postcode = f"{np.random.randint(2000, 7000)}"
    
    # Pick topic based on state prevalence
    topic_probs = topic_state_prevalence[state]
    topic = np.random.choice(topics, p=topic_probs)
    
    # Generate random date
    random_day = np.random.randint(0, days_range)
    date = start_date + timedelta(days=random_day)
    
    # Create document text (simplified)
    document = f"Insurance claim from {city}, {state} regarding {topic.lower()} on {date.strftime('%Y-%m-%d')}"
    
    # Add to lists
    documents.append(document)
    topic_list.append(topic)
    city_list.append(city)
    state_list.append(state)
    lat_list.append(lat)
    lon_list.append(lon)
    postcode_list.append(postcode)
    dates.append(date)

# Create DataFrame
df = pd.DataFrame({
    "document": documents,
    "topic": topic_list,
    "city": city_list,
    "state": state_list,
    "latitude": lat_list,
    "longitude": lon_list,
    "postcode": postcode_list,
    "date": dates
})

print(f"Created sample dataset with {len(df)} documents across {len(topics)} topics")
print(df.head())

# Initialize modeler with the pre-assigned topics
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="document")

# Since we already have topics assigned, we'll manually set document topics
topic_df = df[["document", "topic"]].copy()
topic_df.columns = ["Document", "Topic"]
modeler.topic_assignments = topic_df

# Create geospatial visualizations
print("\nCreating geospatial visualizations...")

# 1. Basic topic map (points on a map)
fig_map = plot_topic_map(
    data=df,
    lat_column="latitude",
    lon_column="longitude",
    topic_column="topic",
    title="Insurance Claims by Topic and Location"
)
fig_map.write_html("topic_map.html")
print("Created basic topic map (saved to topic_map.html)")

# 2. Topic choropleth map by state
# First, need to aggregate by state
state_topic_counts = df.groupby(["state", "topic"]).size().reset_index(name="count")
state_counts = df.groupby("state").size().reset_index(name="total")
state_topic_counts = state_topic_counts.merge(state_counts, on="state")
state_topic_counts["proportion"] = state_topic_counts["count"] / state_topic_counts["total"]

# Create a simple GeoJSON for Australian states (simplified for example)
# For a real application, use a proper GeoJSON file for Australian states
try:
    # Try to load a real Australian states GeoJSON if available
    with open("examples/data/australia_states.json", "r") as f:
        states_geojson = json.load(f)
        print("Loaded Australia states GeoJSON")
except Exception:
    # Create a very simplified version if file not available
    print("Creating simplified Australia states GeoJSON")
    states_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"state": state},
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        np.mean([cities[c]["lon"] for c in cities if cities[c]["state"] == state]),
                        np.mean([cities[c]["lat"] for c in cities if cities[c]["state"] == state])
                    ]
                }
            }
            for state in ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "NT", "ACT"]
        ]
    }

# Create choropleth map for each topic
for topic in topics:
    topic_data = state_topic_counts[state_topic_counts["topic"] == topic]
    topic_dict = dict(zip(topic_data["state"], topic_data["proportion"]))
    
    try:
        fig_choro = plot_topic_choropleth(
            geojson=states_geojson,
            values=topic_dict,
            location_property="state",
            title=f"{topic} Claims by State (Proportion)"
        )
        fig_choro.write_html(f"topic_{topic.replace(' ', '_').lower()}_choropleth.html")
        print(f"Created choropleth map for {topic}")
    except Exception as e:
        print(f"Could not create choropleth map for {topic}: {e}")

# 3. Density heatmap for all topics
fig_density = plot_topic_density_map(
    data=df,
    lat_column="latitude",
    lon_column="longitude",
    topic_column="topic",
    title="Claim Density by Topic"
)
fig_density.write_html("topic_density_map.html")
print("Created topic density map (saved to topic_density_map.html)")

# 4. Space-time heatmap (topics over time and space)
# Aggregate by month and state
df["month"] = df["date"].dt.to_period("M")
spacetime_data = df.groupby(["month", "state", "topic"]).size().reset_index(name="count")
spacetime_data["date"] = spacetime_data["month"].dt.to_timestamp()

try:
    fig_spacetime = plot_topic_spacetime_heatmap(
        data=spacetime_data,
        location_column="state",
        date_column="date",
        topic_column="topic",
        value_column="count",
        title="Topic Distribution Over Time and States"
    )
    fig_spacetime.write_html("topic_spacetime_heatmap.html")
    print("Created space-time heatmap (saved to topic_spacetime_heatmap.html)")
except Exception as e:
    print(f"Could not create space-time heatmap: {e}")

print("\nAll geospatial visualizations complete!")
print("Open the HTML files in a browser to explore the interactive maps.")