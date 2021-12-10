# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_telco():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['MultipleLines'].replace(to_replace='No phone service', value='No', inplace=True)
    for c in df.columns:
        df[c].replace(to_replace='No internet service', value='No', inplace=True)


    # Encoding categorical data
    le = LabelEncoder()

    labelcols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    for c in labelcols:
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values



    encol = [7,14,16]
    ct = ColumnTransformer([("onehot", OneHotEncoder(), encol)], remainder='passthrough')
    X = ct.fit_transform(X)

    y = LabelEncoder().fit_transform(y)


    imp = SimpleImputer(missing_values=np.NAN, strategy= 'mean')
    X[:,25] = imp.fit_transform(X[:,25].reshape(-1,1))[:,0]


    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train[:,:10], X_test[:,:10], y_train, y_test
    # return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    print("Running preprocessing . . .")
    X_train, X_test, y_train, y_test = process_telco()
    print("Finished preprocessing.")
