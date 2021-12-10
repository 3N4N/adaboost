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
        yhat = w[0]
        for i in range(len(example)-1):
            yhat += w[i+1] * example[i]
        return m.tanh(yhat)

    def sgd(self, train, obs):
        w = [0.0 for i in range(len(train[0])+1)]
        for itr in range(self.epoch):
            for i, row in enumerate(train):
                yhat = self.predict(row, w)
                err = obs[i] - yhat
                w[0] = w[0] + self.alpha * err * (1.0 - yhat*yhat)
                for i in range(len(row)):
                    w[i+1] = w[i+1] + self.alpha * err * (1.0 - yhat*yhat) * row[i]
        return w




def main():
    X_train, X_test, y_train, y_test = preproc.process_telco()
    lr = LogisticRegression(0.3, 100)
    w = lr.sgd(X_train, y_train)
    print(w)

    score = 0
    for i, row in enumerate(X_train):
        yhat = lr.predict(row, w)
        pred = 1 if yhat > 0 else -1
        # pred = round(yhat)
        obs = 1 if y_train[i] == 1 else -1
        if pred == obs:
            score += 1

    print("Score:", score/X_train.shape[0])


if __name__ == '__main__':
    main()
