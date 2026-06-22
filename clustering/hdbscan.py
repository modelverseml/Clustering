"""
HDBSCAN (Hierarchical DBSCAN) - work in progress
------------------------------------------------
This file is a stub for an in-progress, from-scratch implementation of
HDBSCAN. The README explains the full algorithm:

    1. Compute the core distance of every point (distance to its
       min_samples-th nearest neighbour).
    2. Build a mutual reachability graph:
           mreach-dist(p, q) = max(core-dist(p), core-dist(q), dist(p, q))
    3. Build a minimum spanning tree over that graph.
    4. Build a cluster hierarchy by progressively removing the largest
       edges, then condense the hierarchy using min_cluster_size.
    5. Pick the most stable clusters across density levels; everything
       else is labelled as noise (-1).

Only the outer scaffolding is sketched below. The interesting steps
(MST construction, cluster condensation, stability scoring) are not yet
implemented. Use scikit-learn's `hdbscan` library for production work.
"""

import numpy as np
import pandas as pd


class HDBSCAN:

    """
    Stub class for a from-scratch HDBSCAN implementation. Not functional yet.
    """

    def __init__(self,data):

        self.data = data
        self.data['is_visited'] = 0


    def get_clusters(self,epsilon : float = 0.2, MinPts: int = 3):

        # Outer scan over all points. The real algorithm would build a
        # minimum spanning tree from the mutual reachability distances and
        # extract stable clusters from the resulting hierarchy.
        for index in self.data.index:

            if self.data.loc[index,'is_visited'] == 0:

                self.data.loc[index,'is_visited'] = 1

                neighbours,core_distance = self.get_neighbours(index)


    def m_reach_distance(self, p_instance, q_instance):

        # mreach-dist(p, q) = max(core-dist(p), core-dist(q), dist(p, q))
        pass

    def get_neighbours(self,index):

        # Should return (neighbour_indices, core_distance) for the given point.
        distances = self.data['p']
