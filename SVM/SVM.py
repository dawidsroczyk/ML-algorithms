import numpy as np
import random
import math

class SVM:

    def __init__(self):
        self.C = 1e-2
        self.tol = 1e-1
        self.max_passes = 100

    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)
        m, n = X.shape
        self.alpha = np.zeros(m)
        self.b = np.zeros(1)
        passes = 0

        
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                Ei = self.pred(X[[i], :]) - y[i]
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C)\
                      or (y[i] * Ei > self.tol and self.alpha[i] > 0):
                    
                    j = i
                    while j == i:
                        j = random.randint(0, m-1)

                    Ej = self.pred(X[[j], :]) - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    L = max([0, alpha_j_old - alpha_j_old]) if y[i] != y[j] \
                        else max(0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old) if y[i] != y[j] \
                        else min(self.C, alpha_i_old + alpha_j_old)
                    
                    if L == H:
                        continue

                    eta = 2 * self.kernel(X[i, :], X[j, :]) - self.kernel(X[i, :], X[i, :]) \
                        - self.kernel(X[j, :], X[j, :])
                    
                    if eta >= 0:
                        continue

                    self.alpha[j] = self.alpha[j] - y[j] * (Ei - Ej) / eta
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    if np.abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] = self.alpha[i] + y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i, :], X[i, :]) \
                        - y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[i, :], X[j, :])
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i, :], X[j, :]) \
                        - y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[j, :], X[j, :])
                    
                    self.b = (b1 + b2) / 2
                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.b = b2
                    
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
                # print(passes)
            else:
                passes = 0

    def pred(self, X):
        res = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            res[j] = np.dot(self.alpha * self.y, self.kernel(self.X, X[j, :])) + self.b
        idx = res >= 0
        res[idx] = 1
        res[~idx] = -1
        # return res, raw_res
        return res
    
    def pred_raw(self, X):
        res = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            res[j] = np.dot(self.alpha * self.y, self.kernel(self.X, X[j, :])) + self.b
        return res
    
    def kernel(self, x1, x2):
        return np.dot(x1, x2)**4
    
    def info(self):
        print(self.alpha)
        print(self.b)