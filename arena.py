import streamlit as st
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

MODEL_METADATA = {
    "RFFA": {"date run": "2020-09-10", "git commit": "4f30914", "model version": "1.0.1"},
    "LSTMv1": {"date run": "2023-01-19", "git commit": "7a36dd0", "model version": "3.1.1"},
    "LSTMv2": {"date run": "2022-07-01", "git commit": "91270e9", "model version": "10.0.1"},
    "WFlow": {"date run": "1975-12-25", "git commit": "f46859d", "model version": "v5_final_2.3_USETHISONE"},
}


st.sidebar.image("https://www.fathom.global/wp-content/uploads/2022/02/FATHOM-LOGO-RGB-1-3.png", width=150)
st.sidebar.title("HydroArena 🌊")
st.sidebar.write("Welcome to HydroArena! This is a tool to compare different hydrological models.")
st.sidebar.write("Select the models you want to compare and the variables/metrics you want to plot.")

models = st.sidebar.multiselect(
    "Select models",
    list(MODEL_METADATA),
    help="Hydrological models to compare",
)

variables = st.sidebar.multiselect(
    "Select variables",
    ["Discharge", "Precipitation", "Temperature"],
    help="Variables to plot for each model"
)

metrics = st.sidebar.multiselect(
    "Select metric",
    ["RMSE", "MAE", "Nash-Sutcliffe"],
    help="Summary metrics to plot for each model"
)

include_map = st.sidebar.checkbox("Include map", help="Include a map with NSE values for Australia")


def generate_rain(num_days, dry_day_prob=0.8, shape=10, scale=0.6):
    is_wet_day = np.random.random(num_days) > dry_day_prob
    
    # Initialize precipitation array with zeros
    precipitation = np.zeros(num_days)
    
    # For wet days, generate precipitation from gamma distribution
    num_wet_days = np.sum(is_wet_day)
    if num_wet_days > 0:
        # Gamma distribution is commonly used for precipitation
        wet_day_amounts = np.random.gamma(shape, scale, size=num_wet_days)
        precipitation[is_wet_day] = wet_day_amounts
    
    return precipitation


def generate_temp(num_days):
    location_params = {
        'mean_temp': 25.0,        # Annual mean temperature in °C
        'annual_amplitude': 10.0,  # Half the difference between summer and winter
        'day_to_day_var': 2.0,     # Day-to-day variability
        'daily_range': 8.0,        # Average daily temperature range
        'phase_shift': 0,          # Days to shift the annual cycle
    }
    
    # Extract parameters
    mean_temp = location_params['mean_temp']
    annual_amplitude = location_params['annual_amplitude']
    day_to_day_var = location_params['day_to_day_var']
    daily_range = location_params['daily_range']
    phase_shift = location_params['phase_shift']
    
    # Create day indices
    days = np.arange(num_days)
    
    # Create the seasonal component (annual cycle)
    # For Northern Hemisphere: minimum around Jan 1 (day 0), maximum around Jul 1 (day 180)
    annual_cycle = mean_temp - annual_amplitude * np.cos(2 * np.pi * (days - phase_shift) / 365.25)
    
    # Add random day-to-day variations (weather patterns)
    # Use cumulative sum of random values to create a random walk with some persistence
    random_daily_changes = np.random.normal(0, 1, num_days)
    # Apply some autocorrelation to simulate weather patterns lasting several days
    weather_pattern = np.zeros(num_days)
    alpha = 0.7  # Autocorrelation coefficient
    weather_pattern[0] = random_daily_changes[0]
    for i in range(1, num_days):
        weather_pattern[i] = alpha * weather_pattern[i-1] + (1-alpha) * random_daily_changes[i]
    
    # Scale the weather pattern to desired day-to-day variability
    weather_pattern = weather_pattern * day_to_day_var
    
    # Combine seasonal component and weather patterns
    daily_mean_temp = annual_cycle + weather_pattern
    
    # Generate daily min and max temperatures
    # The range between min and max varies slightly day to day
    daily_ranges = np.random.normal(daily_range, daily_range/5, num_days)
    daily_min_temp = daily_mean_temp - daily_ranges/2
    daily_max_temp = daily_mean_temp + daily_ranges/2
    
    return daily_mean_temp#, daily_min_temp, daily_max_temp


def generate_discharge(num_days):
    params = {
        'base_flow': 50.0,           # Base flow in m³/s
        'seasonal_amplitude': 30.0,   # Seasonal variation amplitude
        'noise_level': 5.0,           # Random day-to-day variation
        'response_factor': 0.8,       # How strongly discharge responds to precipitation
        'lag_days': 2,                # Lag between precipitation and discharge response
        'memory_days': 10,            # How many days precipitation affects river flow
        'spring_peak_day': 120,       # Day of spring snowmelt peak (early May)
        'spring_peak_magnitude': 1.5, # Relative magnitude of spring peak
        'phase_shift': 0,             # Shift the annual cycle
    }
    
    # Extract parameters
    base_flow = params['base_flow']
    seasonal_amplitude = params['seasonal_amplitude']
    noise_level = params['noise_level']
    response_factor = params['response_factor']
    lag_days = params['lag_days']
    memory_days = params['memory_days']
    spring_peak_day = params['spring_peak_day']
    spring_peak_magnitude = params['spring_peak_magnitude']
    phase_shift = params['phase_shift']
    
    # Create day indices
    days = np.arange(num_days)
    
    # Create the seasonal component (annual cycle)
    # Typically higher in spring and lower in late summer/fall
    seasonal_component = base_flow + seasonal_amplitude * 0.5 * (
        np.sin(2 * np.pi * (days - 90 - phase_shift) / 365.25) + 1)
    
    # Add spring snowmelt peak (typically occurs in late spring)
    # Use a Gaussian curve to model the snowmelt peak
    day_of_year = days % 365
    spring_peak = np.exp(-0.5 * ((day_of_year - spring_peak_day) / 15)**2) * seasonal_amplitude * spring_peak_magnitude
    seasonal_component += spring_peak
    
    # Initialize discharge with seasonal component
    discharge = seasonal_component.copy()
    
    # # Add precipitation response component if precipitation data is provided
    # if precipitation is not None:
    #     # Pad precipitation array for lag calculation
    #     padded_precip = np.zeros(num_days + lag_days + memory_days)
    #     padded_precip[:len(precipitation)] = precipitation
        
    #     # For each day, calculate response to previous days' precipitation
    #     for i in range(num_days):
    #         # Get precipitation from previous days (applying lag)
    #         # Use exponential decay to model how precipitation influence decreases over time
    #         for j in range(memory_days):
    #             decay_factor = np.exp(-0.3 * j)  # Exponential decay
    #             precip_effect = padded_precip[i + lag_days + j] * decay_factor
    #             discharge[i] += precip_effect * response_factor * base_flow / 10
    
    # Add autocorrelated noise to simulate day-to-day variations
    noise = np.zeros(num_days)
    noise[0] = np.random.normal(0, 1)
    alpha = 0.7  # Autocorrelation coefficient
    for i in range(1, num_days):
        noise[i] = alpha * noise[i-1] + (1-alpha) * np.random.normal(0, 1)
    
    # Scale and add noise to discharge
    discharge += noise * noise_level
    
    # Ensure discharge doesn't go below a minimum threshold (1% of base flow)
    discharge = np.maximum(discharge, base_flow * 0.01)
    
    return discharge


def generate_data(model_name, num_days=1000):
    time = pd.date_range("2020-01-01", periods=num_days, freq="D")
    # discharge = np.cumsum(np.random.randn(num_days))
    discharge = generate_discharge(num_days)
    precipitation = generate_rain(num_days)
    temperature = generate_temp(num_days)

    df = pd.DataFrame(
        {
            "Time": time,
            "Discharge": discharge,
            "Precipitation": precipitation,
            "Temperature": temperature,
            "Model": [model_name] * num_days,
        }
    )

    df = df.set_index("Time")
    df["timestamp"] = df.index

    return df

if st.sidebar.button("DO BATTLE! ⚔️") is True:
    with st.spinner(text="Doing battle..."):
        time.sleep(3)
        st.success("Done")
    
    if len(metrics) == 3:
        st.balloons()

    st.header("Battle results 🥊")

    st.subheader("Metadata")

    st.table(MODEL_METADATA)

    if len(models) > 0:
        model_data = pd.concat([generate_data(model) for model in models])

        model_data = pd.concat([model_data, generate_data("Observation")])

        # for variable in variables:
        #     plot = sns.lineplot(data=model_data, x=model_data.index, y=variable, hue="Model")
        #     plt.xticks(rotation=30)
        #     st.pyplot(plot.get_figure())
        #     plt.close()

        st.subheader("Variables")

        for variable in variables:
            fig = px.line(model_data, x="timestamp", y=variable, color="Model")
            st.plotly_chart(fig)

        st.subheader("Metrics")

        metrics_df = data = pd.DataFrame({
            'Model': models,
            'MAE': [6+np.random.random()*4 for _ in range(len(models))],
            'RMSE': [5+np.random.random()*8 for _ in range(len(models))],
            "Nash-Sutcliffe": [.7 + np.random.random()*0.3 for _ in range(len(models))],

        })

        for metric in metrics:
            fig = px.bar(metrics_df, x="Model", y=metric)
            st.plotly_chart(fig)
    
    if include_map is True:
        st.subheader("Mapping")

        cities = {
            'Sydney': (-33.8688, 151.2093),
            'Melbourne': (-37.8136, 144.9631),
            'Brisbane': (-27.4698, 153.0251),
            'Perth': (-31.9505, 115.8605),
            'Adelaide': (-34.9285, 138.6007),
            'Canberra': (-35.2809, 149.1300),
            'Hobart': (-42.8821, 147.3272),
            'Darwin': (-12.4634, 130.8456),
            'Alice Springs': (-23.6980, 133.8807),
            'Gold Coast': (-28.0167, 153.4000)
        }

        # Generate random data for demonstration
        data = []

        for city, (lat, lon) in cities.items():
            # Random value between 0 and 100
            value = np.random.random()
            data.append({
                'City': city,
                'Latitude': lat,
                'Longitude': lon,
                'NSE': value,
                'Size': value / 10  # For bubble size
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create the map using Plotly Express
        fig = px.scatter_mapbox(
            df, 
            lat='Latitude', 
            lon='Longitude',
            color='NSE',
            size='Size',
            hover_name='City',
            hover_data={'NSE': True, 'Latitude': False, 'Longitude': False, 'Size': False},
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=15,
            zoom=3,
            title='Australian Cities - NSE Distribution',
            mapbox_style="carto-positron",  # Light map style
            center={"lat": -25.2744, "lon": 133.7751}  # Center of Australia
        )

        # Add more detailed hover information
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Value: %{marker.color:.1f}'
        )

        # Update layout with more styling options
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='NSE',
                thicknessmode="pixels", 
                thickness=20,
                lenmode="pixels", 
                len=300,
                yanchor="top", 
                y=1,
                ticks="outside"
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=700,
            width=1000
        )

        st.plotly_chart(fig)