#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gradio as gr
import folium
import json
from geopy.geocoders import Nominatim

def generate_map():
    # Attempt to read the latitude, longitude, and service type from a JSON file
    try:
        with open('latest_gesture_data.json', 'r') as f:
            data = json.load(f)
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        service_type = data['service_type']
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Default values if there's an error reading the file or parsing JSON
        lat, lon = 0, 0
        service_type = 'unknown'

    # Initialize the geolocator with a user-agent
    geolocator = Nominatim(user_agent="map_app")
    # Perform reverse geocoding to get location details
    location = geolocator.reverse((lat, lon), exactly_one=True)
    location_details = location.address if location else "Unknown Location"
    
    # Initialize the map with the given coordinates
    map_obj = folium.Map(location=[lat, lon], zoom_start=12,
                         tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                         attr='Esri', name='Esri Satellite')
    
    # Set the marker icon based on the service type
    if service_type == 'Police':
        icon = folium.Icon(icon='shield', prefix='fa', color='blue', icon_color='white')
    elif service_type == 'Ambulance':
        icon = folium.Icon(icon='ambulance', prefix='fa', color='red', icon_color='white')
    elif service_type == 'Emergency':
        icon = folium.Icon(icon='exclamation-triangle', prefix='fa', color='orange', icon_color='white')
    else:
        icon = folium.Icon(color='gray')  # Default icon for unknown or other types
    
    # Add a marker to the map
    folium.Marker([lat, lon], popup=f'{service_type.capitalize()} Service at {location_details}', icon=icon).add_to(map_obj)
    # Add a circle to represent the area
    folium.Circle(
        location=[lat, lon],
        radius=100,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.2
    ).add_to(map_obj)
    
    # Return the HTML representation of the map
    return map_obj._repr_html_()

# Setup the Gradio interface without input fields since it reads from the JSON file
iface = gr.Interface(
    fn=generate_map,
    inputs=[],  # No input fields as data is read from the JSON file
    outputs=gr.HTML(),
    title="Aetos Angelos: A Keen Perception Alert System",
    description="A keen perception alert system that displays real-time data on a map.",
    theme="huggingface"
)

# Launch the interface
iface.launch()

