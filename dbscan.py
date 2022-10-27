from scipy.spatial import KDTree
import numpy as np


def dbscan_dfs(kdtree, X, eps, min_samples, curr, visited, labels, cluster):
    neighbors = kdtree.query_ball_point(X[curr], eps)
    if len(neighbors) <= min_samples:
        return
    labels[curr] = cluster
    for neighbor in neighbors:
        if not visited[neighbor]:
            visited[neighbor] = True
            labels[neighbor] = cluster
            dbscan_dfs(kdtree, X, eps, min_samples, neighbor, visited, labels,
                       cluster)
    return


def dbscan(X, eps=0.5, min_samples=5):
    rows = X.shape[0]
    visited = np.zeros(rows, dtype=np.bool)
    labels = np.zeros(rows)
    kdtree = KDTree(X)
    cluster = 1
    for i in range(rows):
        if not visited[i]:
            visited[i] = True
            dbscan_dfs(kdtree, X, eps, min_samples, i, visited, labels,
                       cluster)
            if labels[i] > 0:
                cluster += 1
    return labels
