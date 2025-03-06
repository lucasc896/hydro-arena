import numpy as np
import pandas as pd


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
        "mean_temp": 25.0,  # Annual mean temperature in °C
        "annual_amplitude": 10.0,  # Half the difference between summer and winter
        "day_to_day_var": 2.0,  # Day-to-day variability
        "daily_range": 8.0,  # Average daily temperature range
        "phase_shift": 0,  # Days to shift the annual cycle
    }

    # Extract parameters
    mean_temp = location_params["mean_temp"]
    annual_amplitude = location_params["annual_amplitude"]
    day_to_day_var = location_params["day_to_day_var"]
    daily_range = location_params["daily_range"]
    phase_shift = location_params["phase_shift"]

    # Create day indices
    days = np.arange(num_days)

    # Create the seasonal component (annual cycle)
    # For Northern Hemisphere: minimum around Jan 1 (day 0), maximum around Jul 1 (day 180)
    annual_cycle = mean_temp - annual_amplitude * np.cos(
        2 * np.pi * (days - phase_shift) / 365.25
    )

    # Add random day-to-day variations (weather patterns)
    # Use cumulative sum of random values to create a random walk with some persistence
    random_daily_changes = np.random.normal(0, 1, num_days)
    # Apply some autocorrelation to simulate weather patterns lasting several days
    weather_pattern = np.zeros(num_days)
    alpha = 0.7  # Autocorrelation coefficient
    weather_pattern[0] = random_daily_changes[0]
    for i in range(1, num_days):
        weather_pattern[i] = (
            alpha * weather_pattern[i - 1] + (1 - alpha) * random_daily_changes[i]
        )

    # Scale the weather pattern to desired day-to-day variability
    weather_pattern = weather_pattern * day_to_day_var

    # Combine seasonal component and weather patterns
    daily_mean_temp = annual_cycle + weather_pattern

    # Generate daily min and max temperatures
    # The range between min and max varies slightly day to day
    daily_ranges = np.random.normal(daily_range, daily_range / 5, num_days)
    daily_min_temp = daily_mean_temp - daily_ranges / 2
    daily_max_temp = daily_mean_temp + daily_ranges / 2

    return daily_mean_temp  # , daily_min_temp, daily_max_temp


def generate_discharge(num_days):
    params = {
        "base_flow": 50.0,  # Base flow in m³/s
        "seasonal_amplitude": 30.0,  # Seasonal variation amplitude
        "noise_level": 5.0,  # Random day-to-day variation
        "response_factor": 0.8,  # How strongly discharge responds to precipitation
        "lag_days": 2,  # Lag between precipitation and discharge response
        "memory_days": 10,  # How many days precipitation affects river flow
        "spring_peak_day": 120,  # Day of spring snowmelt peak (early May)
        "spring_peak_magnitude": 1.5,  # Relative magnitude of spring peak
        "phase_shift": 0,  # Shift the annual cycle
    }

    # Extract parameters
    base_flow = params["base_flow"]
    seasonal_amplitude = params["seasonal_amplitude"]
    noise_level = params["noise_level"]
    response_factor = params["response_factor"]
    lag_days = params["lag_days"]
    memory_days = params["memory_days"]
    spring_peak_day = params["spring_peak_day"]
    spring_peak_magnitude = params["spring_peak_magnitude"]
    phase_shift = params["phase_shift"]

    # Create day indices
    days = np.arange(num_days)

    # Create the seasonal component (annual cycle)
    # Typically higher in spring and lower in late summer/fall
    seasonal_component = base_flow + seasonal_amplitude * 0.5 * (
        np.sin(2 * np.pi * (days - 90 - phase_shift) / 365.25) + 1
    )

    # Add spring snowmelt peak (typically occurs in late spring)
    # Use a Gaussian curve to model the snowmelt peak
    day_of_year = days % 365
    spring_peak = (
        np.exp(-0.5 * ((day_of_year - spring_peak_day) / 15) ** 2)
        * seasonal_amplitude
        * spring_peak_magnitude
    )
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
        noise[i] = alpha * noise[i - 1] + (1 - alpha) * np.random.normal(0, 1)

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
