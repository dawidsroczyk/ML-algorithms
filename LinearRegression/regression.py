import numpy as np

class StochasticGradientDescent:

    def __init__(self, lr, eps=1e-5):
        self.lr = lr
        self.eps = eps
    
    def fit(self, X, y, print_norm=False):
        eps = 0.01

        m = X.shape[0]
        n = 1
        if len(X.shape) > 1:
            n = X.shape[1]

        self.w = np.zeros(shape=(n, 1))

        last = None
        while True:
            last = np.copy(self.w)
            for (xi, yi) in zip(X, y):
                for j in range(n):
                    self.w[j] += self.lr * (yi - self.pred(xi)) * xi[j]
            norm = np.linalg.norm(last - self.w)
            if print_norm:
                print(norm)
            if norm < self.eps:
                break

    def pred(self, X):
        return np.dot(X, self.w)
    
    def info(self):
        return self.w

class BatchGradientDescent:

    def __init__(self, lr, eps=1e-5):
        self.lr = lr
        self.eps = eps
    
    def fit(self, X, y, print_norm=False):

        m = X.shape[0]
        n = 1
        if len(X.shape) > 1:
            n = X.shape[1]

        self.w = np.zeros(shape=(n, 1))
        y = np.copy(y).reshape(m, 1)

        last = None
        while True:
            last = np.copy(self.w)
            for j in range(n):
                self.w[j] += self.lr * np.dot((y - self.pred(X)).reshape(m), X[:, j])
            norm = np.linalg.norm(last - self.w)
            if print_norm:
                print(norm)
            if norm < self.eps:
                break

    def pred(self, X):
        return np.dot(X, self.w)

    def info(self):
        return self.w