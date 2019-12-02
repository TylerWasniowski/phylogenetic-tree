import numpy as np
import cv2 as cv


class TreeNode:
    def __init__(self, name, height=0):
        self.name = name
        self.height = height
        self.left = None
        self.right = None

    def width(self):
        ret = 0
        if self.left is not None:
            ret += 1 + self.left.width()
        if self.right is not None:
            ret += 1 + self.right.width()

        return ret

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


def draw(tree, width=1500, height=750):
    # Make empty image
    img = np.ones((height, width, 3), np.uint8) * 255

    if tree is not None:
        # Draw line at the top
        start_point = (int(width * 0.5), int(0))
        end_point = (int(width * 0.5), int(height * 0.08))
        cv.line(img, start_point, end_point, (0, 0, 0), 2)
        __draw_recurse(img, tree, end_point, int(width * 0.9), int(height * 0.92))

    cv.imshow("Phylogenetic Tree", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def __draw_recurse(img, tree, origin_point, width, height):
    # Draw node name
    name_point = (origin_point[0] + 5, origin_point[1] - 7)
    cv.putText(img, tree.name, name_point, cv.FONT_HERSHEY_SIMPLEX, 1.5, (252, 94, 3), 3)

    # No more children to draw
    if tree.left is None and tree.right is None:
        return

    # Draw horizontal line
    line_width = int(width * 0.5)
    start_point = (origin_point[0] - line_width // 2, origin_point[1])
    end_point = (origin_point[0] + line_width // 2, origin_point[1])
    cv.line(img, start_point, end_point, (0, 0, 0), 2)

    if tree.left is not None:
        left_length = int(height * (1 - tree.left.height / tree.height))
        # Find starting point of left node
        left_origin = (start_point[0], start_point[1] + left_length)
        # Draw vertical line to left node
        cv.line(img, start_point, left_origin, (0, 0, 0), 2)
        # Draw left distance text
        left_dist = tree.height - tree.left.height
        dist_point = (start_point[0] + 5, start_point[1] + left_length // 2)
        cv.putText(img, str(left_dist), dist_point, cv.FONT_HERSHEY_SIMPLEX, 1.2, (104, 123, 252), 2)
        # Draw left node and it's children
        __draw_recurse(img, tree.left, left_origin, width // 2, height - left_length)
    if tree.right is not None:
        right_length = int(height * (1 - tree.right.height / tree.height))
        # Find starting point of right node
        right_origin = (end_point[0], end_point[1] + right_length)
        # Draw vertical line to right node
        cv.line(img, end_point, right_origin, (0, 0, 0), 2)
        # Draw right distance text
        right_dist = tree.height - tree.right.height
        dist_point = (end_point[0] + 5, start_point[1] + right_length // 2)
        cv.putText(img, str(right_dist), dist_point, cv.FONT_HERSHEY_SIMPLEX, 1.3, (104, 123, 252), 2)
        # Draw right node and it's children
        __draw_recurse(img, tree.right, right_origin, width // 2, height - right_length)


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





