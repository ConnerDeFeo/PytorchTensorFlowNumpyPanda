import requests
import pandas as pd

# Step 1, get the data
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 43.1566,
    "longitude": -77.6089,
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max"],
    "timezone": "America/New_York"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["daily"])
df.columns = ["date", "temp_max", "temp_min", "precipitation", "windspeed"]

df.to_csv("data/rochester_weather.csv", index=False)
