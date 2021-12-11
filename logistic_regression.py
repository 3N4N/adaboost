import pandas as pd
import numpy as np
import math as m

import preproc


class LogisticRegression():

    def __init__(self, X, y, alpha=0.3, epoch=int(1e3)):
        self.alpha = alpha
        self.epoch = epoch
        self.X = X
        self.y = y
        self.w = []

    def tanh(self, x):
        return float((m.exp(x) - m.exp(-x)) / (m.exp(x) + m.exp(-x)))

    def predict(self, examples, w=None):
        if w is None:
            w = self.w
        yhat = np.dot(examples, w.reshape(-1,1))
        ret = np.tanh(yhat).flatten()
        ret = (ret + 1) / 2
        return ret

    def sgd(self):
        w = np.array([0.0 for i in range(self.X.shape[1])]) #.reshape(-1,1)
        for itr in range(self.epoch):
            yhat = self.predict(self.X, w)
            err = self.y - yhat
            dloss = self.alpha * (err * (1.0 - yhat*yhat) * self.X.T).mean(axis=1)
            # dloss = self.alpha * (np.matmul(self.X.T, err * (1.0 - yhat*yhat)/2))
            w = w + dloss
            # print(err.shape, self.y.shape, yhat.shape, w.shape, dloss.shape)
        self.w = w.flatten()




def main():
    X_train, X_test, y_train, y_test = preproc.process_telco()
    n_samples, n_features = np.shape(X_train)
    lr = LogisticRegression(X_train, y_train)
    lr.sgd()
    print(lr.w)

    yhat = lr.predict(X_train)

    score = 0
    for i in range(n_samples):
        # pred = 1 if yhat[i] > 0 else -1
        # obs = 1 if y_train[i] == 1 else -1
        pred = round(yhat[i])
        obs = round(y_train[i])
        if pred == obs:
            score += 1
    print("Score:", score/n_samples)

if __name__ == '__main__':
    main()
