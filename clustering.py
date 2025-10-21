import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define the API key and base URL
key = "your_api_key_here"  # Replace with your OpenWeatherMap API key
url = "https://api.openweathermap.org/data/2.5/weather"

# List of cities to analyze
cities = [
    'Helsinki', 'London', 'New York', 'Tokyo', 'Sydney',
    'Paris', 'Berlin', 'Moscow', 'Mumbai', 'Cairo',
    'Toronto', 'Bangkok', 'Oslo', 'Dubai', 'Singapore',
    'Stockholm', 'Amsterdam', 'Copenhagen', 'Reykjavik', 'Athens'
   
]

# Collect weather data for all cities
weather_data = []
print("Fetching weather data...\n")

for city in cities:
    params = {'q': city, 'appid': key, 'units': 'metric'}
    
   
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        data = response.json()

        # Extract the required fields
        city_name = data.get('name', city)
        temp = data['main'].get('temp')
        lat = data.get('coord', {}).get('lat', 0)
            
        weather_data.append({
            'city': city_name,
            'temperature': temp,
            'latitude': abs(lat)  # Distance from equator
        })
    else:
        print(f"{city}: Failed (Status {response.status_code})")
    

# Convert to DataFrame
df = pd.DataFrame(weather_data)

print(f"Data collected for {len(df)} cities")
print("")
print(df.to_string(index=False))
print("\n")


# K-MEANS CLUSTERING
# ============================================================================
print(f"{'='*60}")
print("K-MEANS CLUSTERING: Temperature vs Latitude")
print(f"{'='*60}\n")

# Prepare features for clustering
X = df[['latitude', 'temperature']].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use 3 clusters, it is the optimal number (northern, temperate, tropical)
k = 3
print(f"Applying K-means with {k} clusters...\n")

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Display cluster assignments
print("Cluster Assignments:")
print("")
for cluster_id in range(k):
    cluster_cities = df[df['cluster'] == cluster_id]['city'].tolist()
    cluster_data = df[df['cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} ({len(cluster_cities)} cities):")
    print(f"  Cities: {', '.join(cluster_cities)}")
    print(f"  Average Latitude: {cluster_data['latitude'].mean():.2f}°")
    print(f"  Average Temperature: {cluster_data['temperature'].mean():.2f}°C")

print("\n")


# VISUALIZATION
# ============================================================================
print("Generating visualization...\n")

# Create figure with 1 plot
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot with clusters
scatter = ax.scatter(df['latitude'], df['temperature'], 
                     c=df['cluster'], cmap='viridis', 
                     s=200, alpha=0.6, edgecolors='black', linewidth=1.5)

# Add city labels
for idx, row in df.iterrows():
    ax.annotate(row['city'], 
                (row['latitude'], row['temperature']),
                fontsize=6, ha='right', va='bottom',
                xytext=(-5, 5), textcoords='offset points')

# Add cluster centers
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers_original[:, 0], centers_original[:, 1], 
           c='red', marker='X', s=300, edgecolors='black', 
           linewidth=2, label='Cluster Centers', zorder=5)

ax.set_xlabel('Latitude (Distance from Equator)', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('K-Means Clustering: Latitude vs Temperature', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.show()


print("ANALYSIS COMPLETE")

