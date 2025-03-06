import streamlit as st
import time

import numpy as np
import pandas as pd
import plotly.express as px

from synthetic_data import generate_data

MODEL_METADATA = {
    "RFFA": {
        "date run": "2020-09-10",
        "git commit": "4f30914",
        "model version": "1.0.1",
    },
    "LSTMv1": {
        "date run": "2023-01-19",
        "git commit": "7a36dd0",
        "model version": "3.1.1",
    },
    "LSTMv2": {
        "date run": "2022-07-01",
        "git commit": "91270e9",
        "model version": "10.0.1",
    },
    "WFlow": {
        "date run": "1975-12-25",
        "git commit": "f46859d",
        "model version": "v5_final_2.3_USETHISONE",
    },
}


st.sidebar.image(
    "https://www.fathom.global/wp-content/uploads/2022/02/FATHOM-LOGO-RGB-1-3.png",
    width=150,
)
st.sidebar.title("HydroArena 🌊")
st.sidebar.write(
    "Welcome to HydroArena! This is a tool to compare different hydrological models."
)
st.sidebar.write(
    "Select the models you want to compare and the variables/metrics you want to plot."
)

models = st.sidebar.multiselect(
    "Select models",
    list(MODEL_METADATA),
    help="Hydrological models to compare",
)

variables = st.sidebar.multiselect(
    "Select variables",
    ["Discharge", "Precipitation", "Temperature"],
    help="Variables to plot for each model",
)

metrics = st.sidebar.multiselect(
    "Select metric",
    ["RMSE", "MAE", "Nash-Sutcliffe"],
    help="Summary metrics to plot for each model",
)

include_map = st.sidebar.checkbox(
    "Include map", help="Include a map with NSE values for Australia"
)

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

        st.subheader("Variables")

        for variable in variables:
            fig = px.line(model_data, x="timestamp", y=variable, color="Model")
            st.plotly_chart(fig)

        st.subheader("Metrics")

        metrics_df = data = pd.DataFrame(
            {
                "Model": models,
                "MAE": [6 + np.random.random() * 4 for _ in range(len(models))],
                "RMSE": [5 + np.random.random() * 8 for _ in range(len(models))],
                "Nash-Sutcliffe": [
                    0.7 + np.random.random() * 0.3 for _ in range(len(models))
                ],
            }
        )

        for metric in metrics:
            fig = px.bar(metrics_df, x="Model", y=metric)
            st.plotly_chart(fig)

    if include_map is True:
        st.subheader("Mapping")

        cities = {
            "Sydney": (-33.8688, 151.2093),
            "Melbourne": (-37.8136, 144.9631),
            "Brisbane": (-27.4698, 153.0251),
            "Perth": (-31.9505, 115.8605),
            "Adelaide": (-34.9285, 138.6007),
            "Canberra": (-35.2809, 149.1300),
            "Hobart": (-42.8821, 147.3272),
            "Darwin": (-12.4634, 130.8456),
            "Alice Springs": (-23.6980, 133.8807),
            "Gold Coast": (-28.0167, 153.4000),
        }

        data = []

        for city, (lat, lon) in cities.items():
            value = np.random.random()
            data.append(
                {
                    "City": city,
                    "Latitude": lat,
                    "Longitude": lon,
                    "NSE": value,
                    "Size": value / 10,  # For bubble size
                }
            )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create the map using Plotly Express
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="NSE",
            size="Size",
            hover_name="City",
            hover_data={
                "NSE": True,
                "Latitude": False,
                "Longitude": False,
                "Size": False,
            },
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=15,
            zoom=3,
            title="Australian Cities - NSE Distribution",
            mapbox_style="carto-positron",  # Light map style
            center={"lat": -25.2744, "lon": 133.7751},  # Center of Australia
        )

        # Add more detailed hover information
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>Value: %{marker.color:.1f}"
        )

        # Update layout with more styling options
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="NSE",
                thicknessmode="pixels",
                thickness=20,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                ticks="outside",
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=700,
            width=1000,
        )

        st.plotly_chart(fig)
