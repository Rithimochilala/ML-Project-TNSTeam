import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Cluster mapping
cluster_labels = {
    2: "Casual Listener",
    0: "Music Explorer",
    1: "Heavy Listener"
}

print("===================================")
print("   MUSIC LISTENER SEGMENTATION")
print("===================================")

# Get input from user
listening_hours = float(input("Listening hours per week: "))
songs_per_day = float(input("Songs per day: "))
skip_rate = float(input("Skip rate: "))
playlist_count = float(input("Playlist count: "))

# Keep SAME order as training
features = np.array([[
    listening_hours,
    songs_per_day,
    skip_rate,
    playlist_count
]])

# Scale the new input
scaled_features = scaler.transform(features)

# Predict cluster
cluster = int(model.predict(scaled_features)[0])

# Get meaningful label
segment = cluster_labels[cluster]

print("\n===================================")
print("Prediction Result")
print("===================================")
print("Cluster:", cluster)
print("Listener Segment:", segment)
print("===================================")