import numpy as np


dist_matrix = np.array([
    [np.inf, 6.0, 8.0, 1.0, 2.0, 6.0],
    [6.0, np.inf, 8.0, 6.0, 6.0, 4.0],
    [8.0, 8.0, np.inf, 8.0, 8.0, 8.0],
    [1.0, 6.0, 8.0, np.inf, 2.0, 6.0],
    [2.0, 6.0, 8.0, 2.0, np.inf, 6.0],
    [6.0, 4.0, 8.0, 6.0, 6.0, np.inf],
])

while dist_matrix.shape[0] > 1:
    size = dist_matrix.shape[0]
    min_distance_index = np.argmin(dist_matrix)
    selected = [min_distance_index // size, min_distance_index % size]

    dist_matrix[:, selected[0]] = (dist_matrix[:, selected[0]] + dist_matrix[:, selected[1]]) / 2.0
    dist_matrix[selected[0], :] = (dist_matrix[selected[0], :] + dist_matrix[selected[1], :]) / 2.0

    dist_matrix = np.delete(dist_matrix, selected[1], 0)
    dist_matrix = np.delete(dist_matrix, selected[1], 1)

    print(dist_matrix)





