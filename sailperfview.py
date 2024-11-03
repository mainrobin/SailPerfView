import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime
from meteostat import Point, Hourly
import numpy as np
import altair as alt

# Set page config
st.set_page_config(
    page_title="SailPerfView",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        required_columns = [
            'Record', 'Time', 'Latitude', 'Longitude', 'Speed', 
            'LeanAngle', 'Altitude', 'GForceX', 'GForceZ', 'Lap',
            'GyroX', 'GyroY', 'GyroZ'
        ]
        
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file missing required columns")
            return None
            
        # Convert Time to datetime
        df['Time'] = pd.to_datetime(df['Time'])

        # Convert Speed from km/h to knots (1 km/h = 0.539957 knots)
        df['Speed'] = df['Speed'] * 0.539957
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_weather_data(lat, lon, start_time, end_time):
    """Fetch weather data from Meteostat"""
    try:
        location = Point(lat, lon)
        weather_data = Hourly(location, start_time, end_time)
        data = weather_data.fetch()
        return data
    except Exception as e:
        st.warning(f"Unable to fetch weather data: {str(e)}")
        return None

def create_map_layer(df, current_index):
    """Create PyDeck map layers"""
    # Create path data for the track
    path_data = pd.DataFrame({
        'path': [[[lon, lat] for lon, lat in zip(df['Longitude'], df['Latitude'])]],
    })
    
    # Path layer for the track
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        width_scale=2,
        width_min_pixels=1,
        get_color=[0, 0, 255],
        pickable=True,
        auto_highlight=True
    )
    
    # Scatterplot layer for current position
    current_pos = df.iloc[current_index:current_index+1]
    position_layer = pdk.Layer(
        "ScatterplotLayer",
        data=current_pos,
        get_position=["Longitude", "Latitude"],
        get_color=[255, 0, 0],
        get_radius=5,
        pickable=True
    )
    
    return [path_layer, position_layer]

def create_deck(df, current_index):
    """Create PyDeck map view"""
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    layers = create_map_layer(df, current_index)
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=15,
        pitch=0
    )
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/satellite-v9'
    )
    
    return deck

def create_performance_plot(df, current_index, weather_data=None):
    """Create performance visualization plot using Altair"""
    # Prepare the data for plotting
    plot_df = pd.DataFrame({
        'index': df.index,
        'Speed': df['Speed'],
        'LeanAngle': df['LeanAngle']
    }).melt(id_vars=['index'], var_name='Metric', value_name='Value')
    
    # Create the base line chart
    base = alt.Chart(plot_df).encode(
        x=alt.X('index:Q', title='Time'),
        y=alt.Y('Value:Q', title='Value'),
        color='Metric:N'
    )
    
    # Create the lines
    lines = base.mark_line()
    
    # Create the vertical rule for current position
    rule = alt.Chart(pd.DataFrame({'x': [current_index]})).mark_rule(
        color='red',
        strokeWidth=2
    ).encode(
        x='x:Q'
    )
    
    # Add weather data if available
    if weather_data is not None and not weather_data.empty:
        weather_df = pd.DataFrame({
            'index': df.index,
            'Metric': 'Wind Speed (m/s)',
            'Value': weather_data['wind_speed'].reindex(df.index, method='ffill')
        })
        weather_line = alt.Chart(weather_df).mark_line(
            strokeDash=[5, 5],
            color='red'
        ).encode(
            x=alt.X('index:Q'),
            y=alt.Y('Value:Q'),
            color=alt.value('red')
        )
        chart = (lines + rule + weather_line).properties(
            height=400
        ).configure_axis(
            grid=True
        ).configure_view(
            strokeWidth=0
        )
    else:
        chart = (lines + rule).properties(
            height=400
        ).configure_axis(
            grid=True
        ).configure_view(
            strokeWidth=0
        )
    
    return chart

def main():
    st.title("⛵ SailPerfView")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.data is None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Create and display map
            deck = create_deck(df, st.session_state.current_index)
            st.pydeck_chart(deck)
            
            # Get weather data
            weather_data = get_weather_data(
                df['Latitude'].iloc[0],
                df['Longitude'].iloc[0],
                df['Time'].min(),
                df['Time'].max()
            )
            
            # Create and display performance plot
            chart = create_performance_plot(df, st.session_state.current_index, weather_data)
            st.altair_chart(chart, use_container_width=True)
            
            # Display current metrics
            st.subheader("Current Metrics")
            current_data = df.iloc[st.session_state.current_index]
            cols = st.columns(4)
            cols[0].metric("Speed", f"{current_data['Speed']:.1f} knots")
            cols[1].metric("Lean Angle", f"{current_data['LeanAngle']:.1f}°")
            cols[2].metric("Altitude", f"{current_data['Altitude']:.1f}m")
            cols[3].metric("Lap", int(current_data['Lap']))
            
            # Timeline slider
            st.subheader("Playback Controls")
            current_index = st.slider(
                "Timeline",
                0,
                len(df) - 1,
                st.session_state.current_index
            )
            if current_index != st.session_state.current_index:
                st.session_state.current_index = current_index
                st.rerun()

if __name__ == "__main__":
    main()