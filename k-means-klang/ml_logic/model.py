import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def apply_pca(X_scaled, n_components=24):
    """
    applies PCA transformation.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def cluster_data(X_pca, n_clusters=6):
    """
    applies kmeans clustering to the transformed data with the optimal k of 6.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    return clusters, kmeans
