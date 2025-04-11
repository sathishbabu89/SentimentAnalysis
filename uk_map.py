import folium
import pandas as pd
from streamlit_folium import folium_static
from folium.plugins import HeatMap

def create_uk_sentiment_map(data):
    """Create a UK-wide sentiment heatmap"""
    # Center map on UK
    uk_map = folium.Map(
        location=[54.7024, -3.2766],  # Center of UK
        zoom_start=5,
        tiles="cartodbpositron"
    )
    
    # Prepare heatmap data
    heat_data = []
    for _, row in data.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            # Higher weight for negative sentiment
            weight = (1.5 - row['score']) if row['sentiment'] == 'NEGATIVE' else row['score']
            heat_data.append([row['lat'], row['lon'], weight])
    
    # Add heatmap
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        gradient={'0.3': 'green', '0.5': 'yellow', '0.7': 'red'},
        min_opacity=0.5
    ).add_to(uk_map)
    
    # Add city markers
    for _, row in data.iterrows():
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            color = 'red' if row['sentiment'] == 'NEGATIVE' else 'green'
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=color,
                fill=True,
                popup=f"""
                <b>{row.get('region', 'Location')}</b><br>
                <b>Country:</b> {row.get('country', 'UK')}<br>
                <b>Sentiment:</b> {row['sentiment']}<br>
                <b>Score:</b> {row['score']:.2f}<br>
                <b>Product:</b> {row['product']}<br>
                <i>{row['feedback_text'][:100]}...</i>
                """,
                tooltip=f"{row['region']}: {row['sentiment']}"
            ).add_to(uk_map)
    
    return uk_map