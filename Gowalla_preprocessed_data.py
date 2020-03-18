import pandas as pd
import numpy as np

csv_path = 'Gowalla_Data.csv'
df = pd.read_csv(csv_path)

# add column names
headers = ["user_id", 'Local Time', "Latitude", "Longitude", "Venue_id", "Activity_Category"]
df.columns = headers

# drop 0 values
df = df[df['Latitude'] != 0]
df = df[df['Longitude'] != 0]

# Calculate Z-score
zscore_Lat = (df['Latitude'] - df['Latitude'].mean())/df['Latitude'].std()
zscore_Long = (df['Longitude'] - df['Longitude'].mean())/df['Longitude'].std()

# Remove all rows that have outliers
df = df[(np.abs(zscore_Lat) < 3)]
df = df[(np.abs(zscore_Long) < 3)]

file_name = 'Brightkite_preprocessed_data.csv'
df.to_csv(file_name)

# Now data is ready for prediction