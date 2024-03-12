import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

def load_data(file_path='barrettII_eyes_clustering.xlsx'):
    return pd.read_excel(file_path, sheet_name='Sheet1')

def preprocess_data(df):
    # Create new column
    df['Correct'] = df['Correto']

    # Create dictionary for classifications
    labels_dict = {0 : 'N',
                1 : 'S'}

    # Substitute the names for their labels in the new column
    df['Correct'].replace(list(labels_dict.values()), list(labels_dict.keys()), inplace=True)
    return df

def train_kmeans(X):
    kmeans = KMeans(3, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

df = load_data()
df = preprocess_data(df)

feature_cols = ['AL', 'ACD', 'WTW', 'K1', 'K2']

X = df[feature_cols]

normalizer = StandardScaler()
X_norm = pd.DataFrame(normalizer.fit_transform(X), columns = X.columns)

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

# Train K-Means
kmeans_model = train_kmeans(X_pca)
identified_clusters = kmeans_model.labels_

# Plot the clusters in 2D
unique_clusters = np.unique(identified_clusters)

for cluster_label in unique_clusters:
    cluster_indices = identified_clusters == cluster_label
    plt.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1], label=f'Cluster {cluster_label}')

plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Clusters', loc='best')
plt.savefig('kmeans_clusters.png')

df = X_norm
df['KMeans_Cluster'] = identified_clusters

# Analyze the clusters' characteristics
for cluster_label in np.unique(identified_clusters):
    cluster_data = X[df['KMeans_Cluster'] == cluster_label]
    print("Cluster", cluster_label)
    print(cluster_data.describe())