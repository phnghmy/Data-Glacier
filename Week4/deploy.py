import joblib
import pandas as pd
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('songs.csv')
pd.set_option('display.max_rows', 500)  # Show up to 500 rows
pd.set_option('display.max_columns', 50)  # Show up to 50 columns

# Extract target variable
target = df['Popularity'].values
target2 = target.reshape(-1, 1)

# KMeans clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(target2)

# Save the model
joblib.dump(kmeans, 'kmeans_model.pkl')

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Print the DataFrame to verify
print(df.head())

# Save the DataFrame with clusters
df.to_csv('songs_with_clusters.csv', index=False)

print("Model trained and saved successfully.")