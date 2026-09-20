# ------------------------------------------------------------------
# MUSIC LISTENER SEGMENTATION - MODEL TRAINING SCRIPT
# ------------------------------------------------------------------
# UNSUPERVISED learning project


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans                

df = pd.read_csv('music_listeners.csv')

print("Preview of dataset:")
print(df.head())
print()

X = df[['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)

model.fit(X_scaled)

df['cluster'] = model.labels_

cluster_summary = df.groupby('cluster')[
    ['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']
].mean()

print("Average behaviour per cluster:")
print(cluster_summary)
print()

print("Number of listeners per cluster:")
print(df['cluster'].value_counts())
print()

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully! (model.pkl, scaler.pkl)")