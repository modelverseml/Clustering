"""
K-Medoids clustering (from scratch)
-----------------------------------
Similar to K-Means, but each cluster is represented by an actual data point
(the "medoid") rather than the arithmetic mean of its members. The medoid
of a cluster is the point with the minimum sum of distances to all other
points in the same cluster. Because medoids are real samples, K-Medoids is
more robust to outliers than K-Means.
"""

import numpy as np
from centroid_methods_common_functions import _assign_cluster,_compute_inertia, \
                                _initialise_centorids_or_medoids

class KMedoids:

    """
    Manual implementation of the K-Medoids clustering algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to cluster.
    n_clusters : int
        The number of clusters to form.
    """

    def __init__(self,n_clusters,data):

        self.data = data
        self.n_clusters = n_clusters
        self.inertia_ = np.inf
        self.medoids = _initialise_centorids_or_medoids(self.data,self.n_clusters)

    def get_clusters(self,threshold :float = 0.001 , max_iter:int = 100):

        """
        Perform K-Medoids clustering on the dataset.

        Parameters
        ----------
        threshold : float
            Minimum change in inertia to continue iterations.
        max_iter : int
            Maximum number of iterations to run.

        Returns
        -------
        tuple
            inertia_, cluster_medoids, cluster_labels
        """
        
        for _ in range(max_iter):
            
            ## for all the data check the nearest clusters medoid and assign its cluster to the instance
            cluster_labels = self.data.apply(_assign_cluster,args = (self.medoids,), axis=1)

            # get the inertia for the generated custers labels
            current_inertia = _compute_inertia(self.data,self.medoids,cluster_labels)

    
            if self.inertia_ == np.inf:
                self.inertia_ = current_inertia

            elif abs(self.inertia_ - current_inertia) < threshold:
                break

            self.medoids = self._update_medoids(cluster_labels)
            
            self.inertia_ = current_inertia
  
        final_labels = self.data.apply(_assign_cluster,args = (self.medoids,), axis=1)

        return self.inertia_,self.medoids,final_labels


    def _update_medoids(self,cluster_labels):
        
        """
        Instead of calculating the centroid of the cluster, we compute the sum of distances 
        from each point in the cluster to all other points within the same cluster. 
        The point with the minimum total distance is chosen as the medoid for that cluster.
        """

        medoids = []
        for index in range(self.n_clusters):
            mask = (cluster_labels == index)
            masked_data = self.data.iloc[np.where(mask)[0]].reset_index(drop=True)
            
            distances = [
                np.sum(np.sum((masked_data - instance)**2,axis=1))
                for instance in masked_data.values
                    ]

            medoids.append(masked_data.iloc[np.argmin(distances)].values)

        return medoids