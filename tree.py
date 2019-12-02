import numpy as np


class TreeNode:
    def __init__(self, name, height=0):
        self.name = name
        self.height = height
        self.left = None
        self.right = None

    def __str__(self):
        node_str = self.name
        if self.left is not None:
            node_str += ", (" + str(self.left) + ")" + "-" + str(self.height - self.left.height)
        if self.right is not None:
            node_str += ", (" + str(self.right) + ")" + "-" + str(self.height - self.right.height)

        return node_str


def compute(dist_matrix, render=False):
    # Make a copy so we do not mutate the original
    dist_matrix = np.copy(dist_matrix)
    # Give each node a name: start with A, then B, then C, etc.
    index_to_tree = list(map(lambda i: TreeNode(chr(i + ord('A'))), range(len(dist_matrix))))

    if render:
        print([str(tree) for tree in index_to_tree])
        print(dist_matrix)

    for n in range(len(dist_matrix), 2 * len(dist_matrix) - 1):
        size = dist_matrix.shape[0]
        min_distance_index = np.argmin(dist_matrix)
        selected = [min_distance_index // size, min_distance_index % size]
        # Sort selected so the smaller value is first
        selected = [min(selected), max(selected)]

        # Make new node, with next unused character, and height of half the distance
        new_node_name = chr(n + ord('A'))
        selected_avg_dist = dist_matrix[selected[0], selected[1]] / 2.0
        new_node = TreeNode(new_node_name, selected_avg_dist)
        new_node.left = index_to_tree[selected[0]]
        new_node.right = index_to_tree[selected[1]]

        # Computes new values for a node by averaging the distance to both selected nodes
        # Uses first selected nodes's row and column to save those values
        dist_matrix[:, selected[0]] = (dist_matrix[:, selected[0]] + dist_matrix[:, selected[1]]) / 2.0
        dist_matrix[selected[0], :] = (dist_matrix[selected[0], :] + dist_matrix[selected[1], :]) / 2.0

        # Deletes the second selected node's row and column as it is no longer needed
        dist_matrix = np.delete(dist_matrix, selected[1], 0)
        dist_matrix = np.delete(dist_matrix, selected[1], 1)

        index_to_tree[selected[0]] = new_node
        del index_to_tree[selected[1]]

        if render:
            print([str(tree) for tree in index_to_tree])
            print(dist_matrix)

    return index_to_tree[0]


lecture_data = np.array([
    [np.inf, 6.0, 8.0, 1.0, 2.0, 6.0],
    [6.0, np.inf, 8.0, 6.0, 6.0, 4.0],
    [8.0, 8.0, np.inf, 8.0, 8.0, 8.0],
    [1.0, 6.0, 8.0, np.inf, 2.0, 6.0],
    [2.0, 6.0, 8.0, 2.0, np.inf, 6.0],
    [6.0, 4.0, 8.0, 6.0, 6.0, np.inf],
])

hw_data = np.array([
    [np.inf, 20.0, 60.0, 100.0, 90.0],
    [20.0, np.inf, 50.0, 90.0, 80.0],
    [60.0, 50.0, np.inf, 40.0, 50.0],
    [100.0, 90.0, 40.0, np.inf, 30.0],
    [90.0, 80.0, 50.0, 30.0, np.inf],
])





