"""
Demonstration of Time Series and Geospatial Visualization in Meno

This example shows how to use the extended visualization capabilities in Meno,
including time series, geospatial, and combined time-space visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meno import MenoTopicModeler
import os
import json
import uuid  # For generating claim/customer IDs

# Set up a sample dataset with time, location, and category information
def create_sample_dataset(n_samples=1000, start_date=None, end_date=None, random_seed=42):
    """Create a synthetic dataset with time, location, and topic features."""
    np.random.seed(random_seed)
    
    # Set default date range if not provided
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime(2023, 12, 31)
    
    # Calculate date range in days
    date_range = (end_date - start_date).days
    
    # Generate customer and claim IDs
    n_customers = int(n_samples * 0.4)  # Each customer may have multiple claims
    customer_ids = [f"C{uuid.uuid4().hex[:8]}" for _ in range(n_customers)]
    claim_ids = [f"CLM{uuid.uuid4().hex[:8]}" for _ in range(n_samples)]
    
    # Australian state capitals with approximate coordinates
    locations = {
        "Sydney": {"state": "NSW", "postcode": "2000", "lat": -33.8688, "lon": 151.2093},
        "Melbourne": {"state": "VIC", "postcode": "3000", "lat": -37.8136, "lon": 144.9631},
        "Brisbane": {"state": "QLD", "postcode": "4000", "lat": -27.4698, "lon": 153.0251},
        "Perth": {"state": "WA", "postcode": "6000", "lat": -31.9505, "lon": 115.8605},
        "Adelaide": {"state": "SA", "postcode": "5000", "lat": -34.9285, "lon": 138.6007},
        "Hobart": {"state": "TAS", "postcode": "7000", "lat": -42.8821, "lon": 147.3272},
        "Canberra": {"state": "ACT", "postcode": "2600", "lat": -35.2809, "lon": 149.1300},
        "Darwin": {"state": "NT", "postcode": "0800", "lat": -12.4634, "lon": 130.8456},
    }
    
    # Insurance categories
    categories = ["Home", "Auto", "Health", "Life", "Travel"]
    
    # Synthetic document templates for each category
    templates = {
        "Home": ["Home insurance claim for {damage} damage", 
                "Property damage due to {event}", 
                "House {issue} reported in {location}"],
        "Auto": ["Car accident on {road}", 
                "Vehicle {issue} reported", 
                "Auto insurance claim for {damage}"],
        "Health": ["Health insurance claim for {treatment}", 
                 "Medical coverage inquiry for {condition}", 
                 "Healthcare plan {issue}"],
        "Life": ["Life insurance policy {action}", 
               "Beneficiary {issue} for policy holder", 
               "Term life coverage {action}"],
        "Travel": ["Travel insurance claim for {issue} in {destination}", 
                 "Trip cancellation due to {reason}", 
                 "Overseas medical {issue} claim"],
    }
    
    # Fill-in details
    damages = ["water", "fire", "storm", "structural", "roof", "electrical", "plumbing"]
    events = ["flood", "storm", "fire", "break-in", "vandalism", "accident"]
    issues = ["maintenance", "repair", "damage", "leak", "breakdown", "failure"]
    roads = ["highway", "freeway", "local road", "street", "intersection"]
    treatments = ["surgery", "specialist consultation", "emergency", "hospitalization"]
    conditions = ["chronic condition", "injury", "illness", "preventive care"]
    actions = ["adjustment", "renewal", "cancellation", "inquiry", "update"]
    destinations = ["overseas", "domestic", "international", "remote", "holiday"]
    reasons = ["illness", "weather", "travel restrictions", "emergency", "work"]
    
    # Create sample data
    data = []
    
    for i in range(n_samples):
        # Randomly generate time with more summer incidents (Dec-Feb in Australia)
        days_offset = np.random.choice(range(date_range), p=get_seasonal_weights(date_range))
        date = start_date + timedelta(days=int(days_offset))
        
        # Randomly select location with population-based weights
        location_name = np.random.choice(list(locations.keys()), 
                                        p=[0.25, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        location_info = locations[location_name]
        
        # Add some random variation to coordinates
        lat_jitter = np.random.normal(0, 0.1)
        lon_jitter = np.random.normal(0, 0.1)
        
        # Randomly select category with seasonal weights
        category_weights = get_category_seasonal_weights(date.month)
        category = np.random.choice(categories, p=category_weights)
        
        # Generate synthetic document text
        template = np.random.choice(templates[category])
        text = template.format(
            damage=np.random.choice(damages),
            event=np.random.choice(events),
            issue=np.random.choice(issues),
            road=np.random.choice(roads),
            treatment=np.random.choice(treatments),
            condition=np.random.choice(conditions),
            action=np.random.choice(actions),
            location=location_name,
            destination=np.random.choice(destinations),
            reason=np.random.choice(reasons)
        )
        
        # Generate fake topic assignment (this would normally come from the topic model)
        if category == "Home":
            topic_weights = [0.6, 0.1, 0.1, 0.1, 0.1]  # Higher chance of property topics
        elif category == "Auto":
            topic_weights = [0.1, 0.6, 0.1, 0.1, 0.1]  # Higher chance of vehicle topics
        elif category == "Health":
            topic_weights = [0.1, 0.1, 0.6, 0.1, 0.1]  # Higher chance of medical topics
        elif category == "Life":
            topic_weights = [0.1, 0.1, 0.1, 0.6, 0.1]  # Higher chance of policy topics
        else:  # Travel
            topic_weights = [0.1, 0.1, 0.1, 0.1, 0.6]  # Higher chance of travel topics
            
        topic = np.random.choice(
            ["Property Damage", "Vehicle Incidents", "Medical Claims", 
             "Policy Management", "Travel Issues"],
            p=topic_weights
        )
        
        # Generate a random "importance" value
        importance = np.random.randint(1, 10)
        
        # Assign a customer ID (some customers have multiple claims)
        # Weight towards earlier customer IDs to simulate repeat customers
        customer_idx = min(int(np.random.beta(1.5, 5) * n_customers), n_customers - 1)
        customer_id = customer_ids[customer_idx]
        
        # Create record
        record = {
            "text": text,
            "date": date,
            "location": location_name,
            "state": location_info["state"],
            "postcode": location_info["postcode"],
            "latitude": location_info["lat"] + lat_jitter/10,
            "longitude": location_info["lon"] + lon_jitter/10,
            "category": category,
            "topic": topic,
            "importance": importance,
            "claim_id": claim_ids[i],
            "customer_id": customer_id
        }
        
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

# Helper function to generate seasonal weights for date sampling
def get_seasonal_weights(date_range):
    """Create weights favoring summer months (Dec-Feb in Australia)."""
    weights = np.ones(date_range) / date_range
    
    # Boost summer months (Dec, Jan, Feb)
    for i in range(date_range):
        month = (datetime(2023, 1, 1) + timedelta(days=i)).month
        if month in [12, 1, 2]:  # Summer in Australia
            weights[i] *= 1.5
    
    # Normalize weights
    return weights / weights.sum()

# Helper function to generate seasonal weights for categories
def get_category_seasonal_weights(month):
    """Create seasonal weights for insurance categories based on month."""
    # Base weights
    weights = {
        "Home": 0.3,
        "Auto": 0.3,
        "Health": 0.2,
        "Life": 0.1,
        "Travel": 0.1
    }
    
    # Seasonal adjustments
    if month in [12, 1, 2]:  # Summer
        weights["Home"] *= 1.2  # More storm damage
        weights["Travel"] *= 1.5  # Holiday travel
    elif month in [3, 4, 5]:  # Fall
        weights["Auto"] *= 1.2  # More accidents
        weights["Health"] *= 1.1  # Flu season
    elif month in [6, 7, 8]:  # Winter
        weights["Home"] *= 1.1  # More heating issues
        weights["Health"] *= 1.3  # Winter illnesses
    else:  # Spring
        weights["Life"] *= 1.2  # More policy reviews
        weights["Travel"] *= 1.2  # School holiday travel
    
    # Convert to list and normalize
    weight_list = [weights["Home"], weights["Auto"], weights["Health"], 
                  weights["Life"], weights["Travel"]]
    return np.array(weight_list) / sum(weight_list)

# Australia GeoJSON (simplified for demonstration)
def get_australia_states_geojson():
    """Create a simplified GeoJSON of Australian states."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"state": "NSW"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [141.0, -34.0], [149.5, -34.0], [150.0, -37.5], [141.0, -37.5], [141.0, -34.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "VIC"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [141.0, -34.0], [149.5, -34.0], [150.0, -39.0], [141.0, -39.0], [141.0, -34.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "QLD"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [138.0, -26.0], [153.0, -26.0], [153.0, -29.0], [138.0, -29.0], [138.0, -26.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "SA"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [129.0, -30.0], [141.0, -30.0], [141.0, -38.0], [129.0, -38.0], [129.0, -30.0]
                    ]]
                }
            },
            {
                "type": "Feature", 
                "properties": {"state": "WA"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [113.0, -20.0], [129.0, -20.0], [129.0, -35.0], [113.0, -35.0], [113.0, -20.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "TAS"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [144.0, -40.0], [148.0, -40.0], [148.0, -43.0], [144.0, -43.0], [144.0, -40.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "NT"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [129.0, -12.0], [138.0, -12.0], [138.0, -26.0], [129.0, -26.0], [129.0, -12.0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"state": "ACT"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [148.7, -35.1], [149.4, -35.1], [149.4, -35.8], [148.7, -35.8], [148.7, -35.1]
                    ]]
                }
            }
        ]
    }

def main():
    """Run the time and space visualization demonstration."""
    print("Generating synthetic Australian insurance dataset...")
    df = create_sample_dataset(n_samples=1500)  # Generate more samples for better entity tracking
    
    # Save the dataset
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    df_path = os.path.join(output_dir, "insurance_sample.csv")
    df.to_csv(df_path, index=False)
    print(f"Dataset saved to {df_path}")
    
    # Save the GeoJSON
    geojson = get_australia_states_geojson()
    geojson_path = os.path.join(output_dir, "australia_states.json")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)
    print(f"GeoJSON saved to {geojson_path}")
    
    # Create entity tracking visualizations
    print("\nCreating entity flow visualizations...")
    
    # Sort by customer ID and date for proper transition tracking
    df_sorted = df.sort_values(by=['customer_id', 'date'])
    
    # Create entity transition visualization (Sankey diagram)
    from meno.visualization.enhanced_viz.entity_flow import (
        visualize_entity_transitions, 
        create_entity_flow_sankey, 
        filter_entity_transitions,
        analyze_entity_metrics,
        visualize_entity_metrics
    )
    
    fig_transitions = visualize_entity_transitions(
        df=df_sorted,
        entity_id_column='customer_id',
        topic_column='topic',
        time_column='date',
        min_transition_count=3,  # Filter to transitions that occur at least 3 times
        title="Customer Transitions Between Topics"
    )
    fig_transitions.write_html(os.path.join(output_dir, "customer_topic_transitions.html"))
    
    # Create time-segmented entity flow visualization
    fig_flow = create_entity_flow_sankey(
        df=df_sorted,
        entity_id_column='customer_id',
        topic_column='topic',
        time_column='date',
        value_column='importance',
        time_segments=4,  # Divide the year into 4 quarters
        title="Customer Flow Between Topics Over Time"
    )
    fig_flow.write_html(os.path.join(output_dir, "customer_topic_flow.html"))
    
    # Also create a claim-based flow to show how individual claims move between topics
    # This is useful for tracking how a single claim's categorization may change over time
    df_claims = df_sorted.copy()
    # Add some artificial topic changes for the same claim to simulate recategorization
    claim_updates = []
    for i, claim_id in enumerate(df_claims['claim_id'].unique()[:50]):  # Change topics for some claims
        claim_data = df_claims[df_claims['claim_id'] == claim_id].iloc[0].copy()
        # Create a follow-up entry with a different topic
        claim_data['date'] = claim_data['date'] + timedelta(days=np.random.randint(5, 30))
        all_topics = df_claims['topic'].unique()
        current_topic = claim_data['topic']
        # Pick a different topic
        new_topics = [t for t in all_topics if t != current_topic]
        claim_data['topic'] = np.random.choice(new_topics)
        claim_updates.append(claim_data)
    
    # Add the updates to the dataframe
    df_claims = pd.concat([df_claims, pd.DataFrame(claim_updates)], ignore_index=True)
    df_claims = df_claims.sort_values(by=['claim_id', 'date'])
    
    fig_claim_flow = create_entity_flow_sankey(
        df=df_claims,
        entity_id_column='claim_id',
        topic_column='topic',
        time_column='date',
        time_segments=3,
        title="Claim Recategorization Flow Between Topics"
    )
    fig_claim_flow.write_html(os.path.join(output_dir, "claim_topic_flow.html"))
    
    # Generate advanced entity metrics and visualizations
    print("\nAnalyzing entity flow metrics...")
    
    # Filter to only Auto category for a more focused analysis
    auto_claims = filter_entity_transitions(
        df=df_claims,
        entity_id_column='claim_id',
        topic_column='topic',
        time_column='date',
        categories=['Auto'],
        category_column='category'
    )
    
    # Calculate entity metrics for claims
    claim_metrics = analyze_entity_metrics(
        df=auto_claims,
        entity_id_column='claim_id',
        topic_column='topic',
        time_column='date'
    )
    
    # Create visualizations from the metrics
    claim_figures = visualize_entity_metrics(
        metrics=claim_metrics,
        title_prefix="Auto Claims Analysis"
    )
    
    # Save the metric visualizations
    for name, fig in claim_figures.items():
        fig.write_html(os.path.join(output_dir, f"claim_metrics_{name}.html"))
    
    # Also analyze customer journey metrics for all customers
    customer_metrics = analyze_entity_metrics(
        df=df_sorted,
        entity_id_column='customer_id',
        topic_column='topic',
        time_column='date'
    )
    
    customer_figures = visualize_entity_metrics(
        metrics=customer_metrics,
        title_prefix="Customer Journey Analysis"
    )
    
    # Save the customer metric visualizations
    for name, fig in customer_figures.items():
        fig.write_html(os.path.join(output_dir, f"customer_metrics_{name}.html"))
    
    # Initialize the topic modeler
    modeler = MenoTopicModeler()
    
    # Normally we would preprocess and discover topics, but here we'll use synthetic topics
    modeler.documents = df
    
    # Create time series visualizations
    print("\nCreating time series visualizations...")
    
    # Topic trends over time (line chart)
    fig_line = modeler.visualize_topic_trends(
        time_column="date",
        value_column="importance",
        time_interval="M",
        chart_type="line",
    )
    fig_line.write_html(os.path.join(output_dir, "topic_trends_line.html"))
    
    # Topic composition over time (area chart)
    fig_area = modeler.visualize_topic_trends(
        time_column="date",
        value_column="importance",
        time_interval="M",
        normalize=True,
        chart_type="area",
    )
    fig_area.write_html(os.path.join(output_dir, "topic_composition_area.html"))
    
    # Topic intensity heatmap
    fig_heatmap = modeler.visualize_topic_trends(
        time_column="date",
        value_column="importance",
        time_interval="M",
        chart_type="heatmap",
    )
    fig_heatmap.write_html(os.path.join(output_dir, "topic_intensity_heatmap.html"))
    
    # Topic ridge plot
    fig_ridge = modeler.visualize_topic_trends(
        time_column="date",
        value_column="importance",
        time_interval="M",
        chart_type="ridge",
    )
    fig_ridge.write_html(os.path.join(output_dir, "topic_ridge_plot.html"))
    
    # Create geospatial visualizations
    print("\nCreating geospatial visualizations...")
    
    # Topic map with coordinates
    fig_map = modeler.visualize_geospatial_topics(
        lat_column="latitude",
        lon_column="longitude",
        value_column="importance",
        map_type="point",
    )
    fig_map.write_html(os.path.join(output_dir, "topic_map.html"))
    
    # Topic density map
    fig_density = modeler.visualize_geospatial_topics(
        lat_column="latitude",
        lon_column="longitude",
        map_type="density",
    )
    fig_density.write_html(os.path.join(output_dir, "topic_density_map.html"))
    
    # Topic by postcode
    fig_postcode = modeler.visualize_geospatial_topics(
        postcode_column="postcode",
        value_column="importance",
        map_type="postcode",
    )
    fig_postcode.write_html(os.path.join(output_dir, "topic_postcode_map.html"))
    
    # Load GeoJSON for choropleth
    with open(geojson_path, "r") as f:
        australia_geojson = json.load(f)
    
    # Topic choropleth by state
    fig_choropleth = modeler.visualize_geospatial_topics(
        region_column="state",
        value_column="importance",
        map_type="choropleth",
        geojson=australia_geojson,
        feature_id_property="state",
    )
    fig_choropleth.write_html(os.path.join(output_dir, "topic_choropleth.html"))
    
    # Create time-space visualizations
    print("\nCreating time-space visualizations...")
    
    # Animated map over time
    fig_animated = modeler.visualize_timespace_topics(
        time_column="date",
        lat_column="latitude",
        lon_column="longitude",
        value_column="importance",
        time_interval="M",
        visualization_type="animated_map",
    )
    fig_animated.write_html(os.path.join(output_dir, "topic_animated_map.html"))
    
    # Space-time heatmap
    fig_spacetime = modeler.visualize_timespace_topics(
        time_column="date",
        region_column="state",
        value_column="importance",
        time_interval="M",
        visualization_type="space_time_heatmap",
    )
    fig_spacetime.write_html(os.path.join(output_dir, "topic_spacetime_heatmap.html"))
    
    # Category-time plots
    fig_category = modeler.visualize_timespace_topics(
        time_column="date",
        category_column="category",
        value_column="importance",
        time_interval="M",
        visualization_type="category_time",
        plot_type="line",
    )
    fig_category.write_html(os.path.join(output_dir, "topic_category_line.html"))
    
    fig_category_area = modeler.visualize_timespace_topics(
        time_column="date",
        category_column="category",
        value_column="importance",
        time_interval="M",
        visualization_type="category_time",
        plot_type="area",
    )
    fig_category_area.write_html(os.path.join(output_dir, "topic_category_area.html"))
    
    print("\nVisualization demo complete. Output files saved to:", output_dir)

if __name__ == "__main__":
    main()