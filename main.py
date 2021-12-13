import numpy as np

import preproc
from model.logistic_regression import LogisticRegression
from model.adaboost import Adaboost

def run_logistic_regression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(X_train, y_train)
    lr.sgd()
    print(lr.w)

    print("Train set:")
    yhat = lr.predict(X_train)
    lr.report(np.round(y_train), np.round(yhat))
    print("Test set:")
    yhat = lr.predict(X_test)
    lr.report(np.round(y_test), np.round(yhat))

def run_adaboost(X_train, X_test, y_train, y_test, nitr):
    adb = Adaboost(X_train, y_train, nitr)
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
    X_train, X_test, y_train, y_test = preproc.process_telco(30)
    # X_train, X_test, y_train, y_test = preproc.process_adult(30)
    # X_train, X_test, y_train, y_test = preproc.process_credit(30)
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_adaboost(X_train, X_test, y_train, y_test, 10)
