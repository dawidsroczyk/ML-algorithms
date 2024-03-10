import numpy as np
import pandas as pd
from collections import deque

class Node:

    def __init__(self):
        self.left : Node = None
        self.right : Node = None
        self.compare_function : CompareFunction = None
        self.mean : int = None


class CompareFunction():

    def __init__(self, value, idx : int, label: str):
        self.value : int = value
        self.idx = idx
        self.label = label

    def compare(self, x):
        return x[self.idx] >= self.value

    def __str__(self):
        return f'{self.label}'


class RegressionTree:

    def __init__(self, alpha: int) -> None:
        self.root = Node()
        self.alpha = alpha
        self.col_count = None


    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        
        assert type(X) is pd.DataFrame
        assert type(Y) is pd.Series
        assert len(X.shape) == 2
        assert len(Y.shape) == 1
        assert X.shape[0] == Y.shape[0]

        row_count = X.shape[0]
        col_count = X.shape[1]
        self.col_count = col_count
        X_cols = X.columns

        X_np = X.to_numpy()
        Y_np = Y.to_numpy()

        todo_nodes = deque()
        todo_nodes.append(self.root)

        node_indexes = {}
        node_indexes[self.root] = np.full(row_count, True)
        
        while len(todo_nodes) > 0:

            node = todo_nodes.pop()
            node_idx = node_indexes[node]

            X_node = X_np[node_idx]
            Y_node = Y_np[node_idx]

            node_idx_sorted = np.argsort(X_node, axis=0)
            X_node_sorted = np.take_along_axis(X_node, node_idx_sorted, axis=0)
            Y_node_sorted = np.take_along_axis(np.repeat(Y_node, col_count, axis=0).reshape((X_node.shape[0], col_count)), node_idx_sorted, axis=0)

            errors = np.zeros((X_node.shape[0]+1, col_count))

            for s in range(X_node.shape[0]+1):

                Y_1 = Y_node_sorted[:s]
                Y_2 = Y_node_sorted[s:]

                err = np.sum(np.power(Y_1 - np.average(Y_1, axis=0), 2), axis=0) + np.sum(np.power(Y_2 - np.average(Y_2, axis=0), 2), axis=0)
                errors[s] = err
            
            min_row, min_col = divmod(np.argmin(errors), errors.shape[1])
            critical_point = X_node_sorted[min_row, min_col]

            node.compare_function = CompareFunction(critical_point, min_col, f'{X_cols[min_col]} >= {critical_point}')

            smaller_node = Node()
            node_indexes[smaller_node] = np.logical_and(node_idx, X_np[:, min_col] < critical_point)
            smaller_node.mean = np.mean(Y_np[node_indexes[smaller_node]])

            greater_node = Node()
            node_indexes[greater_node] = np.logical_and(node_idx, X_np[:, min_col] >= critical_point)
            greater_node.mean = np.mean(Y_np[node_indexes[greater_node]])

            c1, c2 = np.count_nonzero(node_indexes[smaller_node]), np.count_nonzero(node_indexes[greater_node])

            if c1 == 0 or c2 == 0:
                node.compare_function = None
                continue

            node.left = smaller_node
            node.right = greater_node


            if (c1 >= self.alpha or c2 >= self.alpha):
                
                todo_nodes.append(smaller_node)
                todo_nodes.append(greater_node)


    def pred(self, X):
        
        assert self.col_count is not None
        assert len(X.shape) == 2
        assert self.col_count == X.shape[1]

        res = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            node : Node = self.root
            while node.left is not None or node.right is not None:
                if node.compare_function.compare(x):
                    node = node.right
                else:
                    node = node.left
            res[i] = node.mean
        
        return res
