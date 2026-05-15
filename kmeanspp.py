"""
K-Means++ clustering (from scratch)
-----------------------------------
Same iterative refinement as K-Means, but the initial centroids are seeded
with the K-Means++ procedure: pick the first centroid at random, then pick
each subsequent centroid with probability proportional to the squared
distance from the closest already-chosen centroid. This spreads the initial
seeds out and usually gives better, more stable clusters than random init.
"""

from kmeans import ManualKMeans
import numpy as np
from centroid_methods_common_functions import _initialise_centorids_or_medoids

class ManualKMeansPP:

    """
    A manual implementation of the K-Means ++  clustering algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to cluster.
    n_clusters : int
        The number of clusters to form.

    Attributes
    ----------
    cluster_centroids : np.ndarray
        Array of cluster centroids.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    """

    def __init__(self,n_clusters, data):
        
        self.data = data
        self.n_clusters = n_clusters


    def get_clusters(self,threshold :float = 0.001 , max_iter:int = 100):

        manual_kmeans_algo = ManualKMeans(data = self.data,n_clusters = self.n_clusters)

        manual_kmeans_algo.cluster_centroids = _initialise_centorids_or_medoids(self.data,self.n_clusters)

        return manual_kmeans_algo.get_clusters(threshold,max_iter)

    



