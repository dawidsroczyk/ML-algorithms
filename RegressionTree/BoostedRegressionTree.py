import numpy as np
import pandas as pd
import RegressionTree as rt


class BoostedRegressionTree:

    def __init__(self, learning_rate : float = 0.01,
                 n_estimators : int = 100, max_samples_leaf : int = 30,
                 max_depth : int = 10) -> None:
        
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_samples_leaf = max_samples_leaf
        self.max_depth = max_depth

        self.model_coef = np.ones(self.n_estimators)
        self.models = []
    


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

        f_0 : rt.RegressionTree = rt.RegressionTree(max_samples_leaf = self.max_samples_leaf,
                                                    max_depth = self.max_depth)
        f_0.fit(X, Y)
        self.model_coef[0] = 1
        self.models.append(f_0)

        X_np = X.to_numpy()
        Y_np = Y.to_numpy()

        for i in range(1, self.n_estimators):
            
            f_x = self.pred(X)
            r_i = 2.0 * (Y_np - f_x) # / row_count

            h_i = rt.RegressionTree(max_samples_leaf = self.max_samples_leaf,
                                    max_depth = self.max_depth)
            r_i_series = pd.Series(r_i)
            h_i.fit(X, r_i_series)
            # return r_i

            h_i_x = h_i.pred(X_np)
            # choosing this lam will minimize train error in this iteration
            lam = np.sum(np.multiply(h_i_x, Y_np - f_x)) / np.sum(np.multiply(h_i_x, h_i_x))

            self.models.append(h_i)
            # self.model_coef[i] = lam
            self.model_coef[i] = self.learning_rate



    def pred(self, X, head = None) -> None:
        
        assert self.col_count is not None
        assert len(X.shape) == 2
        assert self.col_count == X.shape[1]

        if head is None:
            head = self.n_estimators
        
        if head < 0:
            head = 0
        
        head = min(len(self.models), head)

        result = np.zeros(X.shape[0])

        for i in range(head):

            model = self.models[i]
            result_i = model.pred(X.to_numpy())
            result = result + self.model_coef[i] * result_i
        
        return result

