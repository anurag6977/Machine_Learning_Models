import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# KMeans Clustering
from sklearn.cluster import KMeans
class KMeansClustering:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.centroids = None

    def fit(self):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

    def show_results(self):
        print("Final Assignment")
        print("{:<10} {:<5} {:<5} {:<10}".format("Dataset No", "X", "Y", "Assignment"))
        for i, (x, y) in enumerate(self.data):
            print("{:<10} {:<5} {:<5} {:<10}".format(i + 1, x, y, self.labels[i] + 1))  # Adding 1 to labels for 1-based indexing

        print("\nCluster Centroids:")
        for i, centroid in enumerate(self.centroids):
            print(f"Cluster {i + 1}: {centroid}")

        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(self.n_clusters):
            cluster_points = self.data[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=f'Cluster {i + 1}')

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, color='yellow', edgecolors='black', marker='X', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()


    

