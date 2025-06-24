import os
import pandas as pd

# Create a directory to store the data
if not os.path.exists('data'):
    os.makedirs('data')

# Download the March 2024 Green Taxi data
df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet')

# Save the data to a local file
df.to_parquet('data/green_tripdata_2024-03.parquet')

# Get the number of rows
num_rows = df.shape[0]
print(f"Sample Data: {df.head(5)}")
print(f"Number of rows: {num_rows}")