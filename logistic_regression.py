import numpy as np
import pandas as pd
import math as m
from sklearn.metrics import confusion_matrix as cm

import preproc


class LogisticRegression():

    def __init__(self, X, y, alpha=0.3, epoch=1e3):
        self.alpha = alpha
        self.epoch = int(epoch)
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
        n_samples, n_features = self.X.shape
        w = np.array([0.0 for i in range(n_features)])
        for itr in range(self.epoch):
            yhat = self.predict(self.X, w)
            # mismatch = 0
            # for i in range(n_samples):
            #     if round(yhat[i]) != self.y[i]:
            #         mismatch += 1
            # mismatch /= n_samples
            # if (mismatch < 0.5):
            #     break
            dloss = ((self.y - yhat) * (1.0 - yhat*yhat) * self.X.T).mean(axis=1)
            w = w + self.alpha * dloss
            # print((self.y - yhat).shape, self.y.shape, yhat.shape, w.shape, dloss.shape)
        self.w = w.flatten()


def report(y, y_pred):
    [[TN, FP], [FN, TP]] = cm(y, y_pred)
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    ppv = TP / (TP + FP)
    fdr = FP / (FP + TP)
    f1s = 2*TP / (2*TP + FP + FN)
    print(TP, TN, FP, FN)
    print("TPR:", tpr)
    print("TNR:", tnr)
    print("PPV:", ppv)
    print("FDR:", fdr)
    print("F1-Score:", f1s)


def run_logistic_regression():
    # X_train, X_test, y_train, y_test = preproc.process_telco(0)
    X_train, X_test, y_train, y_test = preproc.process_adult(0)
    # X_train, X_test, y_train, y_test = preproc.process_credit(0)
    lr = LogisticRegression(X_train, y_train)
    lr.sgd()
    print(lr.w)

    print("Train set:")
    yhat = lr.predict(X_train)
    report(np.round(y_train), np.round(yhat))
    print("Test set:")
    yhat = lr.predict(X_test)
    report(np.round(y_test), np.round(yhat))



if __name__ == '__main__':
    run_logistic_regression()
