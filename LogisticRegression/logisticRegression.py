import numpy as np

class StochasticGradientDescent:

    def __init__(self, lr=1e-3, eps=1e-5):
        self.lr = lr
        self.eps = eps
    
    def fit(self, X, y, print_norm=False):
        
        m, n = X.shape
        self.w = np.zeros(n)

        while True:
            last = self.w.copy()

            for(xi, yi) in zip(X, y):
                for j in range(n):
                    self.w[j] += self.lr * (yi - self.pred(xi)) * xi[j]

            norm = np.linalg.norm(self.w - last)
            if print_norm:
                print(norm)
            if norm < self.eps:
                break

    def pred(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def info(self):
        return self.w