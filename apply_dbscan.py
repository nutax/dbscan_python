from dbscan import dbscan
from collections import defaultdict
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from kneed import KneeLocator

files = [f'data/{x}' for x in os.listdir('data/')]
eps = [3, 125, 125]
minPts = [4, 3, 3]
names = ['cerebellum', 'colon', 'endometrium',
         'hippocampus', 'kidney', 'liver', 'placenta']

for file, e, m in zip(files, eps, minPts):
    data = np.genfromtxt(file, delimiter=',')
    y, X = data[:, 0], data[:, 1:]
    clusters = dbscan(X, e, m)
    cluster_y_fr = defaultdict(lambda: defaultdict(int))
    cluster_sum_fr = defaultdict(int)
    for cluster, label in zip(clusters, y):
        cluster_y_fr[cluster][label] += 1
        cluster_sum_fr[cluster] += 1
    cluster_max_fr = {}
    cluster_max_label = {}
    errors = 0
    for cluster, labels in cluster_y_fr.items():
        cluster_max_label[cluster] = max(labels, key=labels.get)
        cluster_max_fr[cluster] = labels[cluster_max_label[cluster]]
        errors += cluster_sum_fr[cluster] - cluster_max_fr[cluster]
    with open(f"dbscan_results/{file.split('/')[1].split('.')[0]}.txt", 'w') as out:
        for cluster in cluster_sum_fr:
            out.write(
                f'( {cluster_max_fr[cluster]} / {cluster_sum_fr[cluster]} ) -> {names[int(cluster_max_label[cluster])]}\n')
        out.write(f'Mismatches: {round(100*errors/len(X))}%')
