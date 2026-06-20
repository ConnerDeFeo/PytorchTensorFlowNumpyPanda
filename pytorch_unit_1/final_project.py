import pandas as pd
import matplotlib.pyplot as plt

# Import the data
df = pd.read_csv("data/rochester_weather.csv")
df["date"] = pd.to_datetime(df["date"])

# Create the target value
df["windspeed_tomorrow"] = df["windspeed"].shift(-1)
df = df.dropna()

# Plot the data to take a look at it
plt.figure(figsize=(10, 30))
plt.show()