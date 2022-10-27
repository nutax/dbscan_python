import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

files = [f'data/{x}' for x in os.listdir('data/')]


for file in files:
    data = np.genfromtxt(file, delimiter=',')
    y, X = data[:, 0], data[:, 1:]
    minNeigbors = X.shape[1]  # (Sander et al., 1998)
    nearest_neighbors = NearestNeighbors(n_neighbors=minNeigbors)
    # (Nadia Rahmah and Imas Sukaesih Sitanggang, 2016)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, minNeigbors - 1], axis=0)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig(
        f"distance_analysis/{file.split('/')[1].split('.')[0]}.png", dpi=300)
