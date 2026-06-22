"""
Shared helpers for centroid-based clustering algorithms
-------------------------------------------------------
These functions are reused by K-Means, K-Means++, Mini-Batch K-Means++ and
K-Medoids:

    _assign_cluster                - assign a sample to its nearest centroid
    _update_centroids              - move centroids to the mean of their cluster
    _compute_inertia               - within-cluster sum of squared distances
    _initialise_centorids_or_medoids - K-Means++ probabilistic seeding
"""

import numpy as np
import pandas as pd

def _assign_cluster(instance,centroids) -> int:

    ## Assign a data point to the nearest cluster centroid.
    distances = [np.sum((centroid - instance.values) **2) for centroid in centroids]
    return np.argmin(distances)


def _update_centroids(data,n_clusters,clusters_labels) -> None:
    
    ## Update centroids based on the mean of points in each cluster.
    centroids = []

    for cluster_idx in range(n_clusters):
        cluster_mask = pd.Series(clusters_labels,index = data.index) == cluster_idx
        centroids.append(data[cluster_mask].mean(axis=0).values)

    return  np.array(centroids)


def _compute_inertia(data,centroids,clusters_labels) -> float:
    
    ## Compute the sum of squared distances to the nearest centroid.

    inertia = 0

    for cluster_idx,centroid in enumerate(centroids):

        cluster_mask = pd.Series(clusters_labels, index = data.index) == cluster_idx

        # Compute squared distance of each point to centroid and sum over all dimensions + rows
        inertia += ((data[cluster_mask] -  centroid)**2).sum(axis=1).sum() 

    return inertia

def _initialise_centorids_or_medoids(data,n_clusters) -> list:

    ''' 
    K-Means++ improves K-Means by choosing better initial centroids. It starts by picking 
    one random centroid, then calculates distances from this centroid to all other 
    data points. Points farther away have a higher chance of being selected as the next 
    centroid. This probability-based selection continues until K centroids are chosen, 
    helping to spread them out and improve clustering accuracy.
    '''

    cluster_centroids = [data.sample(n=1).values]

    for _ in range(n_clusters-1):

        distances = np.array([
            min(
                np.sum((center-instance)**2)
                for center in cluster_centroids
            )
            for instance in data.values
        ])

        probabilites = distances/distances.sum()

        cluster_centroids.append(data.sample(n=1,weights = probabilites).values)


    return cluster_centroids
