"""
Energy Consumption Forecasting with Transformers
"""

import pandas as pd

# load .txt file
df = pd.read_csv("household_power_consumption.txt",
                 sep= ";",
                 low_memory=False,
                 na_values="?")
print(df.head())

# combine date and time in one object as "datetime"
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                format= "%d/%m/%Y %H:%M:%S")

print(df.head())

# select related columns and remove rows that includes NULL values
df = df[["datetime", "Global_active_power"]].dropna()


# set datetime as index
df = df.set_index("datetime")

print(df.head())

# convert minute to hour
df = df.resample("1h").mean()

df = df.ffill()

# save the clean data
df.to_csv("cleaned_power.csv")
