import numpy as np
import math

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.prob = [{}, {}]
        self.p0 = y[y == '0'].shape[0] / y.shape[0]
        self.p1 = y[y == '1'].shape[0] / y.shape[0]

        for xi in X:
            for word in xi.split(' '):
                word = self.clear_signs(word)
                self.prob[0][word] = 0
                self.prob[1][word] = 0
        
        self.zero_count = 0
        self.one_count = 0

        for xi, yi in zip(X, y):
            yi = int(yi)
            if yi == 1:
                self.one_count += len(xi.split(' '))
            else:
                self.zero_count += len(xi.split(' '))

            for word in xi.split(' '):
                word = self.clear_signs(word)
                self.prob[yi][word] += 1
        
        V = X.shape[0]
        self.zero_count += V
        self.one_count += V

        for word in self.prob[0]:
            self.prob[0][word] += 1
            self.prob[0][word] /= self.zero_count
            self.prob[1][word] += 1
            self.prob[1][word] /= self.one_count
        print((self.p0, self.p1))

    
    def pred(self, X):
        res = np.zeros(X.shape[0])
        idx = 0
        for xi in X:
            p0 = 0
            p1 = 1
            for word in xi.split(' '):
                word = self.clear_signs(word)
                if self.prob[0][word] > 0 and self.prob[1][word] > 0:
                    p0 = p0 + math.log(self.prob[0][word])
                    p1 = p1 + math.log(self.prob[1][word])
            if p0 > p1:
                res[idx] = 0
            else:
                res[idx] = 1
            idx += 1
        return res
            
                

    def clear_signs(self, word):
        word = word.replace('.', '')
        word = word.replace(',', '')
        word = word.replace(':', '')
        word = word.replace('\'', '')
        word = word.replace('-', '')
        return word.lower()
                
        