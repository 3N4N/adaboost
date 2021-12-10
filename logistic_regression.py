import pandas as pd
import numpy as np
import math as m

import preproc


class LogisticRegression():

    def __init__(self, alpha, epoch):
        self.alpha = alpha
        self.epoch = epoch

    def tanh(self, x):
        return float((m.exp(x) - m.exp(-x)) / (m.exp(x) + m.exp(-x)))

    def predict(self, example, w):
        yhat = np.dot(example, w.reshape(-1,1))
        return np.tanh(yhat).flatten()

    def sgd(self, x, y):
        w = np.array([0.0 for i in range(x.shape[1])]) #.reshape(-1,1)
        for itr in range(self.epoch):
            yhat = self.predict(x, w)
            err = y - yhat
            dloss = self.alpha * (err * (1.0 - yhat*yhat) * x.T).mean(axis=1)
            w = w + dloss
            # w = w + self.alpha * (err * (1.0 - yhat*yhat) * x.T).mean(axis=1)
            # print(err.shape, y.shape, yhat.shape, w.shape, dloss.shape)
        return w.flatten()




def main():
    X_train, X_test, y_train, y_test = preproc.process_telco()
    n_obs = X_train.shape[0]
    lr = LogisticRegression(0.3, 100)
    w = lr.sgd(X_train, y_train)
    print(w)

    yhat = lr.predict(X_train, w)
    print(yhat.shape)

    score = 0
    for i in range(n_obs):
        pred = 1 if yhat[i] > 0 else -1
        obs = 1 if y_train[i] == 1 else -1
        if pred == obs:
            score += 1
    print("Score:", score/n_obs)

if __name__ == '__main__':
    main()
