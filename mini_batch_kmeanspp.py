"""
Mini-Batch K-Means++ clustering (from scratch)
----------------------------------------------
A scalable variant of K-Means++. Instead of using the whole dataset to
update centroids on every iteration, a random mini-batch is drawn at each
step and used to update the centroids. This dramatically reduces compute
per iteration on large datasets while converging to a similar solution.
"""

import numpy as np
from centroid_methods_common_functions import _initialise_centorids_or_medoids,\
                                                _assign_cluster,_compute_inertia, \
                                                _update_centroids

class MiniBatchKmeanspp:

    """
    Manual implementation of Mini-Batch K-Means++.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to cluster.
    n_clusters : int
        The number of clusters to form.
    """

    def __init__(self,data,n_clusters):

        self.data = data
        self.n_clusters = n_clusters
        self.inertia_ = np.inf
        self.cluster_centroids = _initialise_centorids_or_medoids(self.data,self.n_clusters)

    
    def get_clusters(self, batch_size,threshold : float = 0.01, max_iter:int = 100):

        """
        The only differnece between Kmeans ++ and mini batch is 
        dataset consideration while calcluating centorids, for Kmeans++ we take full dataset
        where as we take batch wise in each iteration in mini batch - mostly used for large datasets

        """

        for _ in range(max_iter):

            batch_index = np.random.choice(len(self.data), batch_size, replace=False)

            batch_data = self.data.iloc[batch_index]

            cluster_labels = self.data.apply(_assign_cluster,args = (self.cluster_centroids,), axis=1)

            # get the inertia for the generated custers labels
            current_inertia = _compute_inertia(batch_data,self.cluster_centroids,cluster_labels)


            if self.inertia_ == np.inf:
                self.inertia_ = current_inertia


            elif abs(self.inertia_ - current_inertia) < threshold:
                break

            self.cluster_centroids = _update_centroids(batch_data,self.n_clusters,cluster_labels)
            
            self.inertia_ = current_inertia

        final_labels = self.data.apply(_assign_cluster,args = (self.cluster_centroids,), axis=1)

        return self.inertia_,self.cluster_centroids,final_labels