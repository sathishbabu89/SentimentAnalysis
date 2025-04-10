import folium
from streamlit_folium import folium_static
import pandas as pd
from folium.plugins import HeatMap

def create_london_sentiment_map(data):
    """
    Create an interactive folium map showing sentiment heatmap in London.
    
    Args:
        data (pd.DataFrame): Data containing sentiment analysis with London locations
        
    Returns:
        folium.Map: Configured map object
    """
    # Create base map centered on London
    m = folium.Map(
        location=[51.5074, -0.1278], 
        zoom_start=11,
        tiles='cartodbpositron'
    )
    
    # Prepare heatmap data
    heat_data = []
    for _, row in data.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            # Weight negative sentiments more heavily
            weight = (1 - row['score']) if row['sentiment'] == 'NEGATIVE' else row['score']
            heat_data.append([row['lat'], row['lon'], weight])
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        gradient={'0.4': 'blue', '0.6': 'lime', '0.8': 'red'},
        max_zoom=13
    ).add_to(m)
    
    # Add markers for individual points
    for _, row in data.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            color = 'red' if row['sentiment'] == 'NEGATIVE' else 'green'
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"<b>{row.get('borough', 'Location')}</b><br>"
                      f"Sentiment: {row['sentiment']}<br>"
                      f"Score: {row['score']:.2f}<br>"
                      f"<i>{row['feedback_text'][:100]}...</i>",
                tooltip=f"{row['sentiment']} ({row['score']:.2f})"
            ).add_to(m)
    
    return m
