"""
K-Means clustering (from scratch)
---------------------------------
Partitions the data into K clusters by alternating two steps:

    1. Assign each sample to its nearest centroid.
    2. Move each centroid to the mean of its assigned samples.

Iteration stops when the change in inertia (within-cluster sum of squares)
falls below a threshold or when max_iter is reached.
"""

import numpy as np
from centroid_methods_common_functions import _assign_cluster,_compute_inertia,_update_centroids
class ManualKMeans:

    """
    A manual implementation of the K-Means clustering algorithm.

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

    def __init__(self,data : np.array,n_clusters : int) -> None :

        self.data = data
        self.n_clusters = n_clusters
        self.cluster_centroids: np.ndarray = np.array([])
        self.inertia_: float = np.inf

        ## Before assignning clusters first we need inintial centroids
        self._initialize_centroids()



    def get_clusters(self,threshold :float = 0.001 , max_iter:int = 100):

        """
        Perform K-Means clustering on the dataset.

        Parameters
        ----------
        threshold : float
            Minimum change in inertia to continue iterations.
        max_iter : int
            Maximum number of iterations to run.

        Returns
        -------
        tuple
            inertia_, cluster_centroids, cluster_labels
        """
        
        for _ in range(max_iter):
            
            ## for all the data check the nearest clusters center and assign its cluster to the instance
            cluster_labels = self.data.apply(_assign_cluster,args = (self.cluster_centroids,), axis=1)

            # get the inertia for the generated custers labels
            current_inertia = _compute_inertia(self.data,self.cluster_centroids,cluster_labels)


            if self.inertia_ == np.inf:
                self.inertia_ = current_inertia


            elif abs(self.inertia_ - current_inertia) < threshold:
                break

            self.cluster_centroids = _update_centroids(self.data,self.n_clusters,cluster_labels)
            
            self.inertia_ = current_inertia
  
        final_labels = self.data.apply(_assign_cluster,args = (self.cluster_centroids,), axis=1)

        return self.inertia_,self.cluster_centroids,final_labels
            

    def _initialize_centroids(self) -> None:

        ## Randomly select initial centroids from the data.
        self.cluster_centroids =  self.data.sample(n=self.n_clusters).values





