"""
OPTICS (Ordering Points To Identify the Clustering Structure)
-------------------------------------------------------------
A density-based clustering algorithm that improves on DBSCAN by ordering
the points so that spatially-close points become neighbours in the
ordering. The output is a reachability plot rather than a flat clustering,
so different clustering granularities can be extracted afterwards by
picking different reachability thresholds.

Key distances used:
    core-dist(p)         - distance to p's MinPts-th nearest neighbour
    reach-dist(p, q)     - max(core-dist(p), dist(p, q))

A priority queue (min-heap keyed by reachability distance) is used to
process points in order of how easily they can be reached from already
processed core points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq

class OPTICS:

    """
    Manual implementation of the OPTICS clustering algorithm.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to cluster.
    epsilon : float
        Maximum neighborhood radius (only used to bound the neighbour search).
    MinPts : int
        Minimum points required to compute a core distance.
    cluster_epsilon : float
        Reachability threshold used when extracting flat clusters from the
        ordering produced by OPTICS.
    """

    def __init__(self,data,epsilon,MinPts,cluster_epsilon):

        self.data = data
        self.epsilon = epsilon
        self.MinPts = MinPts
        self.cluster_epsilon = cluster_epsilon

        self.cols_to_use = data.columns.tolist()
        self.order_dataset = []

        self.data['is_visited'] = 0

    def order_data(self):

        for index in self.data.index:

            if self.data.loc[index,'is_visited'] == 0:
                self.data.loc[index,'is_visited'] = 1

                neighbours_index,core_distance = self._get_neighbours(index)

                self.order_dataset.append((0, index))

                if core_distance !=np.inf:

                    self._expand_cluster(index, neighbours_index,core_distance)



        reachability_distances = [dist for dist, idx in self.order_dataset]

        # plt.figure(figsize=(10,4))
        # plt.bar(range(len(reachability_distances)), reachability_distances, color='skyblue')
        # plt.xlabel("Points in OPTICS order")
        # plt.ylabel("Reachability Distance")
        # plt.title("OPTICS Reachability Plot")
        # plt.show()


        return self.extract_clusters(self.order_dataset, self.cluster_epsilon) 
    

    def extract_clusters(self,order_dataset, eps_cluster):

        """
        order_dataset: list of (reachability_distance, index)
        eps_cluster: threshold to separate clusters
        """
        clusters = {}
        cluster_id = -1
        current_cluster = []

        for reachability, idx in order_dataset:
            if reachability <= eps_cluster:
                if cluster_id == -1:
                    cluster_id += 1
                current_cluster.append(idx)
            else:
                if current_cluster:
                    # Assign cluster_id to all points in current cluster
                    for i in current_cluster:
                        clusters[i] = cluster_id
                    cluster_id += 1
                    current_cluster = []

        # Assign remaining points
        for i in current_cluster:
            clusters[i] = cluster_id

        # Convert to pandas Series (aligned with original DataFrame)
        n_points = max(idx for _, idx in order_dataset) + 1
        cluster_series = pd.Series([clusters.get(i, -1) for i in range(n_points)])
        return cluster_series
    

    def _expand_cluster(self,index,neigbours_index,core_distance):

        seeds = []

        self._update_neighbours(seeds,index, neigbours_index, core_distance)

        
        while(seeds):

            reachable_distance,neigbour_index = heapq.heappop(seeds)

            if neigbour_index and self.data.loc[neigbour_index,'is_visited'] == 0:
                
                self.data.loc[neigbour_index,'is_visited'] = 1

                self.order_dataset.append((reachable_distance,neigbour_index))
                
                neighbour_neighbours_indices,neighbour_core_distance = self._get_neighbours(neigbour_index)

                if neighbour_core_distance != np.inf:

                    self._update_neighbours(seeds,neigbour_index, neighbour_neighbours_indices, neighbour_core_distance)

    
    def _update_neighbours(self,seeds, core_index, neighbours_index, core_distance):

        for neighbour_index in neighbours_index:

            if self.data.loc[neighbour_index,'is_visited'] == 0:

                dis = np.linalg.norm(
                    self.data.loc[core_index, self.cols_to_use].values -
                    self.data.loc[neighbour_index, self.cols_to_use].values
                )

                reachable_distance = max(dis,core_distance)
                
                heapq.heappush(seeds, (reachable_distance, neighbour_index))


    def _get_neighbours(self,index):

        instance = self.data.loc[index,self.cols_to_use].values

        distances = np.linalg.norm(self.data[self.cols_to_use].values - instance, axis=1)

        mask = distances <= self.epsilon
        
        neighbour_distances = distances[mask]

        neighbour_indices = self.data[mask].index.tolist()

        # Compute core distance
        if len(neighbour_distances) >= self.MinPts:
            core_distance = np.sort(neighbour_distances)[self.MinPts - 1]

        elif len(neighbour_distances) > 0:
            core_distance = np.max(neighbour_distances)
        else:
            core_distance = np.inf

        return neighbour_indices, core_distance
    

def heap_push(heap,item):
        
    inserted = False

    for i, (dis,index) in enumerate(heap):

        if dis > item[0]:
            heap.insert(i,item)
            inserted = True
            break

    if not inserted:
        heap.append(item)

def heap_pop(heap):

    if heap:
        return heap.pop(0)
    return None