"""Clustering algorithms implemented from scratch.

A collection of unsupervised clustering algorithms written from scratch with
NumPy/pandas, grouped by family:

    Centroid / partition : ManualKMeans, ManualKMeansPP, MiniBatchKmeanspp, KMedoids
    Density              : DBSCAN, OPTICS, HDBSCAN (stub)
    Hierarchical         : Agglomerative, Divisive, BIRCH
    Distribution         : GMM
    Other                : MeanShift, AffinityPropagation, SpectralClustering

Each algorithm returns cluster labels; see README.md for the theory and
examples/walkthrough.py for a runnable demo that checks them against
scikit-learn.
"""

from .affinity_propagation import AffinityPropagation
from .agglomerative import Agglomerative
from .birch import BIRCH
from .dbscan import DBSCAN
from .divisive import Divisive
from .gmm import GMM
from .hdbscan import HDBSCAN
from .kmeans import ManualKMeans
from .kmeanspp import ManualKMeansPP
from .kmedoids import KMedoids
from .mean_shift import MeanShift
from .mini_batch_kmeanspp import MiniBatchKmeanspp
from .optics import OPTICS
from .spectral import SpectralClustering

__all__ = [
    "ManualKMeans",
    "ManualKMeansPP",
    "MiniBatchKmeanspp",
    "KMedoids",
    "DBSCAN",
    "OPTICS",
    "HDBSCAN",
    "Agglomerative",
    "Divisive",
    "BIRCH",
    "GMM",
    "MeanShift",
    "AffinityPropagation",
    "SpectralClustering",
]

__version__ = "1.0.0"
