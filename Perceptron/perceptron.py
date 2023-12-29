import numpy as np

class Perceptron:

    def __init__(self, eta=0.01, n_iter=50,
                 random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    
    def fit(self, X, Y):
        
        # add ones to features
        X1 = np.ones((X.shape[0], 1))
        X = np.hstack((X, X1))

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        for i in range(self.n_iter):
            for xi, target in zip(X, Y):
                # pred = self.predict(xi)
                # self.w += self.eta * xi * (target - pred)
                update = self.eta * (target - self.predict(xi))
                self.w += update * xi
        
        # return self
    
    def info(self):
        print(f"w: {self.w}")

    def net_input(self, X):
        return np.dot(X, self.w)
    
    def predict(self, X):
        # print(self.net_input(X))
        return np.where(self.net_input(X) >= 0.0, 1, 0)