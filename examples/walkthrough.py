"""End-to-end clustering walkthrough.

Replaces the old notebook. Runs every from-scratch algorithm in the package on
small standardised synthetic datasets and reports how good each clustering is:

    - ARI  (Adjusted Rand Index) vs the ground-truth labels: 1.0 = perfect,
      0.0 = random. Lets us check the algorithm recovered the real groups.
    - Silhouette: how tight/separated the clusters are, using the data only
      (no ground truth). Higher is better.

Each algorithm has a slightly different API, so they are wrapped in a small
registry of (name, dataset, run-function). Anything that errors is reported
and the walkthrough continues.

Run from the repository root:

    python examples/walkthrough.py            # print the results table
    python examples/walkthrough.py --plot     # also save cluster scatter plots
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from clustering import (  # noqa: E402  (import after the sys.path tweak)
    AffinityPropagation,
    Agglomerative,
    BIRCH,
    DBSCAN,
    Divisive,
    GMM,
    KMedoids,
    ManualKMeans,
    ManualKMeansPP,
    MeanShift,
    MiniBatchKmeanspp,
    SpectralClustering,
)

RANDOM_STATE = 42
N_SAMPLES = 300
GENERATED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_images"
)


def make_datasets():
    """Three standardised 2-D datasets with known ground-truth labels."""
    scaler = StandardScaler()

    def make(generator, **kwargs):
        X, y = generator(**kwargs)
        X = scaler.fit_transform(X)
        return pd.DataFrame(X, columns=["x1", "x2"]), y

    blobs = make(make_blobs, n_samples=N_SAMPLES, centers=3, cluster_std=0.6,
                 random_state=RANDOM_STATE)
    moons = make(make_moons, n_samples=N_SAMPLES, noise=0.08, random_state=RANDOM_STATE)
    circles = make(make_circles, n_samples=N_SAMPLES, noise=0.05, factor=0.4,
                   random_state=RANDOM_STATE)
    return {"blobs": blobs, "moons": moons, "circles": circles}


def quality(labels, y_true, X):
    """Return (ARI vs ground truth, silhouette). Silhouette needs >= 2 clusters."""
    labels = np.asarray(labels)
    ari = adjusted_rand_score(y_true, labels)

    valid = labels != -1  # ignore DBSCAN/OPTICS noise points for silhouette
    if valid.sum() >= 2 and len(set(labels[valid])) >= 2:
        sil = silhouette_score(np.asarray(X)[valid], labels[valid])
    else:
        sil = float("nan")
    return ari, sil


# Each entry: label -> (dataset key, function(df) -> cluster labels).
# The lambdas hide each algorithm's slightly different API behind one call.
ALGORITHMS = {
    "KMeans (blobs)":            ("blobs",  lambda df: ManualKMeans(df.copy(), n_clusters=3).get_clusters()[2]),
    "KMeans++ (blobs)":          ("blobs",  lambda df: ManualKMeansPP(n_clusters=3, data=df.copy()).get_clusters()[2]),
    "MiniBatch KMeans++ (blobs)":("blobs",  lambda df: MiniBatchKmeanspp(df.copy(), n_clusters=3).get_clusters(batch_size=60)[2]),
    "KMedoids (blobs)":          ("blobs",  lambda df: KMedoids(n_clusters=3, data=df.copy()).get_clusters()[2]),
    "GMM (blobs)":               ("blobs",  lambda df: GMM(df.copy(), n_components=3, random_state=RANDOM_STATE).fit().get_clusters()),
    "Agglomerative (blobs)":     ("blobs",  lambda df: Agglomerative(df.copy(), n_clusters=3, linkage="average").get_clusters()),
    "Divisive (blobs)":          ("blobs",  lambda df: Divisive(df.copy(), n_clusters=3).get_clusters()),
    "BIRCH (blobs)":             ("blobs",  lambda df: BIRCH(df.copy(), n_clusters=3, threshold=0.5).fit().get_clusters()),
    "MeanShift (blobs)":         ("blobs",  lambda df: MeanShift(df.copy(), bandwidth=1.0).fit().get_clusters()),
    "AffinityProp (blobs)":      ("blobs",  lambda df: AffinityPropagation(df.copy(), damping=0.9, max_iter=200).fit().get_clusters()),
    "DBSCAN (moons)":            ("moons",  lambda df: DBSCAN(df.copy(), epsilon=0.3, MinPts=5).get_clusters().values),
    "Agglomerative (moons)":     ("moons",  lambda df: Agglomerative(df.copy(), n_clusters=2, linkage="single").get_clusters()),
    "Spectral (circles)":        ("circles", lambda df: SpectralClustering(df.copy(), n_clusters=2, sigma=0.3, random_state=RANDOM_STATE).fit().get_clusters()),
}


def run_all(datasets):
    """Run every algorithm, collecting labels and quality metrics."""
    results = []
    for name, (ds_key, fn) in ALGORITHMS.items():
        df, y_true = datasets[ds_key]
        try:
            labels = np.asarray(fn(df))
            ari, sil = quality(labels, y_true, df)
            results.append({
                "algorithm": name,
                "n_clusters": len(set(labels[labels != -1])),
                "ARI": round(ari, 3),
                "silhouette": round(sil, 3),
                "labels": labels,
                "dataset": ds_key,
            })
        except Exception as exc:  # keep going if one implementation trips
            results.append({
                "algorithm": name, "n_clusters": None, "ARI": None,
                "silhouette": None, "labels": None, "dataset": ds_key,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return results


def print_results(results):
    table = pd.DataFrame([
        {k: r.get(k) for k in ("algorithm", "dataset", "n_clusters", "ARI", "silhouette")}
        for r in results
    ])
    print(table.to_string(index=False))

    errors = [r for r in results if r.get("error")]
    if errors:
        print("\nAlgorithms that did not complete:")
        for r in errors:
            print(f"  {r['algorithm']}: {r['error']}")


def save_plots(results, datasets):
    """Scatter-plot each successful clustering, coloured by predicted label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(GENERATED_DIR, exist_ok=True)
    ok = [r for r in results if r.get("labels") is not None]

    n = len(ok)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, r in zip(axes, ok):
        df, _ = datasets[r["dataset"]]
        ax.scatter(df["x1"], df["x2"], c=r["labels"], cmap="tab10", s=10)
        ax.set_title(f"{r['algorithm']}\nARI={r['ARI']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(ok):]:
        ax.axis("off")

    fig.tight_layout()
    out = os.path.join(GENERATED_DIR, "clustering_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved cluster plots to {out}")


def main():
    parser = argparse.ArgumentParser(description="Run the clustering walkthrough.")
    parser.add_argument("--plot", action="store_true", help="save cluster scatter plots")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print("Clustering walkthrough  (ARI vs ground truth, higher = better)")
    print("=" * 70)
    datasets = make_datasets()
    results = run_all(datasets)
    print_results(results)

    if args.plot:
        save_plots(results, datasets)

    print()


if __name__ == "__main__":
    main()
