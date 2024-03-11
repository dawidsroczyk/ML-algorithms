import numpy as np
import pandas as pd
from collections import deque

class Node:

    def __init__(self):
        # left child
        self.left : Node = None
        # right child
        self.right : Node = None
        # function by which data is split
        self.compare_function : CompareFunction = None
        # mean of the elements belonging to gis node
        self.mean : int = None


'''
This class's goal is to represent a function lambda x: x[self.idx] >= self.value
where self.value is specified by the user
'''
class CompareFunction():

    def __init__(self, value, idx : int, label: str):
        # value by which to compare
        self.value : int = value
        # index of x by which to compare
        self.idx = idx
        # how to print this function
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
        
        # just some assertions about the data
        assert type(X) is pd.DataFrame
        assert type(Y) is pd.Series
        assert len(X.shape) == 2
        assert len(Y.shape) == 1
        assert X.shape[0] == Y.shape[0]

        # save some info about data
        row_count = X.shape[0]
        col_count = X.shape[1]
        self.col_count = col_count
        X_cols = X.columns

        # convert data to numpy array
        X_np = X.to_numpy()
        Y_np = Y.to_numpy()

        # initialize "stack"
        todo_nodes = deque()
        todo_nodes.append(self.root)

        # for every node in a tree save array of booleans
        # depicting which training samples land there
        node_indexes = {}
        node_indexes[self.root] = np.full(row_count, True)
        
        # consider all nodes
        while len(todo_nodes) > 0:
            
            # get top node
            node = todo_nodes.pop()
            # get boolean values of training samples belonging to this node
            node_idx = node_indexes[node]

            # get training samples belonging to this node
            X_node = X_np[node_idx]
            Y_node = Y_np[node_idx]

            # sort every columns in X_node and save indexes
            node_idx_sorted = np.argsort(X_node, axis=0)
            X_node_sorted = np.take_along_axis(X_node, node_idx_sorted, axis=0)
            # for every column in X sort Y seperately according to node_idx_sorted
            Y_node_sorted = np.take_along_axis(np.repeat(Y_node, col_count, axis=0).reshape((X_node.shape[0], col_count)), node_idx_sorted, axis=0)

            # save RRS to this array
            errors = np.zeros((X_node.shape[0]+1, col_count))

            # for every split point
            for s in range(X_node.shape[0]+1):

                # split Y_node_sorted into two parts
                Y_1 = Y_node_sorted[:s]
                Y_2 = Y_node_sorted[s:]

                # calculate RRS error and save it
                err = np.sum(np.power(Y_1 - np.average(Y_1, axis=0), 2), axis=0) + np.sum(np.power(Y_2 - np.average(Y_2, axis=0), 2), axis=0)
                errors[s] = err
            
            # get best feature to split and best point to split it
            min_row, min_col = divmod(np.argmin(errors), errors.shape[1])
            # calculate split point in that feature
            critical_point = X_node_sorted[min_row, min_col]

            # define compare function
            node.compare_function = CompareFunction(critical_point, min_col, f'{X_cols[min_col]} >= {critical_point}')

            # create new left node
            smaller_node = Node()
            node_indexes[smaller_node] = np.logical_and(node_idx, X_np[:, min_col] < critical_point)
            smaller_node.mean = np.mean(Y_np[node_indexes[smaller_node]])

            # create new right node
            greater_node = Node()
            node_indexes[greater_node] = np.logical_and(node_idx, X_np[:, min_col] >= critical_point)
            greater_node.mean = np.mean(Y_np[node_indexes[greater_node]])

            # count number of samples in each children node
            c1, c2 = np.count_nonzero(node_indexes[smaller_node]), np.count_nonzero(node_indexes[greater_node])

            # check if one of the splits is empty
            if c1 == 0 or c2 == 0:
                node.compare_function = None
                continue
            
            # assign new childred
            node.left = smaller_node
            node.right = greater_node

            # if splits are too large
            if (c1 >= self.alpha or c2 >= self.alpha):
                # split them further
                todo_nodes.append(smaller_node)
                todo_nodes.append(greater_node)


    def pred(self, X):
        
        assert self.col_count is not None
        assert len(X.shape) == 2
        assert self.col_count == X.shape[1]

        res = np.zeros(X.shape[0])

        # for every sample in X
        for i, x in enumerate(X):

            # iterate to the leaf
            node : Node = self.root
            while node.left is not None or node.right is not None:
                if node.compare_function.compare(x):
                    node = node.right
                else:
                    node = node.left
            
            # assign result value
            res[i] = node.mean
        
        return res
