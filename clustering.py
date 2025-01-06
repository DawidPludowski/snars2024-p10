from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def get_graph(path: str | Path) -> tuple[nx.Graph, NDArray[np.float64]]:
    data = pd.read_csv(path, header=None)
    m = np.array(data)
    G = nx.Graph(m)
    m = m / m.sum(axis=0).reshape(-1, 1)

    return G, m


def get_eigenvecs(
    m: NDArray[np.float64], max_coordinates: int = -1
) -> NDArray[np.float64]:
    _, vec = np.linalg.eig(m)
    vec = vec.real
    vec = vec[:, :max_coordinates]
    return vec


def get_clustering(
    eigen_repr: NDArray[np.float64], n_clusters: int = 4
) -> NDArray[np.int64]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    res = model.fit_predict(eigen_repr)
    return res


def draw_clustering(G: nx.Graph, clustering: NDArray[np.int64]) -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    n_clusters = np.unique(clustering).shape[0]
    pallete = sns.color_palette("tab10", n_clusters)

    color_map = color_map = [pallete[x] for x in clustering]
    nx.draw(G, node_color=color_map, ax=ax)
    return fig


def grid_find_params(
    eigen_repr: NDArray[np.float64],
    max_cooridantes_grid: list[int],
    n_cluster_grid: list[int],
    score_name: str = "silhouette",
) -> tuple[int, int]:
    best_score = -1
    best_max_cooridinates, best_n_clusters = -1, -1
    for max_cooridinates in max_cooridantes_grid:
        for n_clusters in n_cluster_grid:
            clustering = get_clustering(
                eigen_repr[:, :max_cooridinates], n_clusters=n_clusters
            )
            if score_name == "silhouette":
                score = silhouette_score(
                    eigen_repr[:, :max_cooridinates], labels=clustering
                )
            else:
                score_name = calinski_harabasz_score(
                    eigen_repr[:, :max_cooridinates], labels=clustering
                )
            if score > best_score:
                best_score = score
                best_max_cooridinates = max_cooridinates
                best_n_clusters = n_clusters

    return best_max_cooridinates, best_n_clusters
