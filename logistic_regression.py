import pandas as pd
import numpy as np
import math as m

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
            mismatch = 0
            for i in range(n_samples):
                if round(yhat[i]) != self.y[i]:
                    mismatch += 1
            mismatch /= n_samples
            # if (mismatch < 0.5):
            #     break
            dloss = ((self.y - yhat) * (1.0 - yhat*yhat) * self.X.T).mean(axis=1)
            w = w + self.alpha * dloss
            # print((self.y - yhat).shape, self.y.shape, yhat.shape, w.shape, dloss.shape)
        self.w = w.flatten()




def main():
    # X_train, X_test, y_train, y_test = preproc.process_telco(10)
    X_train, X_test, y_train, y_test = preproc.process_adult(10)
    lr = LogisticRegression(X_train, y_train)
    lr.sgd()
    print(lr.w)

    n_samples, n_features = np.shape(X_train)
    yhat = lr.predict(X_train)
    score = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n_samples):
        if round(yhat[i]) == round(y_train[i]):
            score += 1
            if round(yhat[i]) == 1:
                tp += 1
            else:
                tn += 1
        else:
            if round(yhat[i]) == 1:
                fp += 1
            else:
                fn += 1
    score /= n_samples * 100
    print("Train set:")
    print("Score:", score)
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

    n_samples, n_features = np.shape(X_test)
    yhat = lr.predict(X_test)
    score = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n_samples):
        if round(yhat[i]) == round(y_test[i]):
            score += 1
            if round(yhat[i]) == 1:
                tp += 1
            else:
                tn += 1
        else:
            if round(yhat[i]) == 1:
                fp += 1
            else:
                fn += 1
    score /= n_samples * 100
    print("Test set:")
    print("Score:", score)
    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)


if __name__ == '__main__':
    main()
