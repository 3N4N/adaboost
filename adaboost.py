import pandas as pd
import numpy as np
import math as m

import preproc
from logistic_regression import LogisticRegression


class Adaboost():
    def __init__(self, X, y, n_clf):
        self.n_clf = n_clf
        self.X = X
        self.y = y
        self.clfs = []
        self.z = []

    def fit(self):
        self.n_clf = 5

        n_samples, n_features = np.shape(self.X)
        w = np.full(n_samples, (1 / n_samples))
        examples = np.c_[self.X, self.y]

        for k in range(self.n_clf):
        # while len(self.clfs) < self.n_clf:
            samples = examples[np.random.choice(range(n_samples), (n_samples,), replace=True, p=w),:]
            _X = samples[:, :-1]
            _y = samples[:, -1]
            n_resamples = _X.shape[0]
            clf = LogisticRegression(_X, _y)
            clf.sgd()
            yhat = np.round(clf.predict(_X, clf.w))
            error = 0
            cnt = 0
            for j in range(n_samples):
                if yhat[j] != _y[j]:
                    cnt += 1
                    error += w[j]
            print(error, cnt)
            if error > 0.5:
                continue
            for j in range(n_resamples):
                if yhat[j] == _y[j]:
                    w[j] *= (error/(1-error))
            w /= w.sum()
            self.clfs.append(clf)
            _z = np.log(1-error) - np.log(error)
            self.z.append(_z)

        self.z = np.divide(self.z, np.sum(self.z))
        print(self.z)
        self.n_clf = len(self.clfs)

    def predict(self, X, y):
        yhat = []
        for k in range(self.n_clf):
            yhat.append(self.z[k] * np.round(self.clfs[k].predict(X)))
        yhat = np.round(np.sum(np.array(yhat), axis=0))

        return yhat

def run_adaboost():
    # X_train, X_test, y_train, y_test = preproc.process_telco(0)
    X_train, X_test, y_train, y_test = preproc.process_adult(None)
    # X_train, X_test, y_train, y_test = preproc.process_credit(0)
    adb = Adaboost(X_train, y_train, 5)
    adb.fit()

    n_samples, n_features = np.shape(X_train)
    yhat = adb.predict(X_train, y_train)
    score = 0
    for i in range(n_samples):
        if round(yhat[i]) == round(y_train[i]):
            score += 1
    score /= n_samples / 100
    print("Train score:", score)

    n_samples, n_features = np.shape(X_test)
    yhat = adb.predict(X_test, y_test)
    score = 0
    for i in range(n_samples):
        if round(yhat[i]) == round(y_test[i]):
            score += 1
    score /= n_samples / 100
    print("Test score:", score)


if __name__ == '__main__':
    run_adaboost()
