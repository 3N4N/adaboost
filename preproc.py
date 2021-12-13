import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_telco(percentile=None):
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['MultipleLines'].replace(to_replace='No phone service', value='No', inplace=True)
    for c in df.columns:
        df[c].replace(to_replace='No internet service', value='No', inplace=True)


    labelcols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    onehotcols = [ 'InternetService', 'Contract', 'PaymentMethod' ]

    for c in labelcols:
        df[c] = LabelEncoder().fit_transform(df[c])

    for c in onehotcols:
        df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)
    df.drop(columns=onehotcols, axis=1, inplace=True)

    imp = SimpleImputer(missing_values=np.NAN, strategy='mean')
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)

    X = df.drop(columns=['Churn']).to_numpy()
    y = df['Churn'].to_numpy()

    if percentile is not None and percentile != 0:
        X = SelectPercentile(mutual_info_classif, percentile=percentile).fit_transform(X, y)

    # print(np.sum(y == 1))
    # print(np.sum(y == 0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc_X = StandardScaler()
    # sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test  = np.c_[X_test,  np.ones(X_test.shape[0])]

    return X_train, X_test, y_train, y_test

def process_adult(percentile=None):

    columns = [
            'age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country', 'salary'
    ]
    numeric_cols = [
            'age', 'fnlwgt', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week'
    ]
    yes_no_colms = ['sex', 'salary']
    yes_no_map = [{'Male': 1, 'Female': 0}, {'>50K': 1, '<=50K': 0}]
    labelcols = [cat for cat in columns[:-1] if cat not in numeric_cols]
    labelcols = [cat for cat in labelcols if cat not in yes_no_colms]

    df = pd.read_csv('data/adult/adult.data', header=None, names=columns)
    df_test = pd.read_csv('data/adult/adult.test', header=None, names=columns).drop(0)
    n_samples_train = df.shape[0]
    n_samples_test = df_test.shape[0]
    df = df.append(df_test)
    df.columns = columns

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.replace('?', np.NAN)

    imp = SimpleImputer(missing_values=np.NAN, strategy='most_frequent')
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)

    for c in labelcols:
        df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)
    df.drop(columns=labelcols, axis=1, inplace=True)

    for i in range(len(yes_no_colms)):
        df[yes_no_colms[i]].replace(yes_no_map[i], inplace=True)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c])

    X = df.drop(columns=['salary']).to_numpy()
    y = df['salary'].to_numpy()

    if percentile is not None and percentile != 0:
        X = SelectPercentile(mutual_info_classif, percentile=percentile).fit_transform(X, y)

    X_train = X[0:n_samples_train,:]
    y_train = y[0:n_samples_train]
    X_test = X[n_samples_train:,:]
    y_test = y[n_samples_train:]

    sc_X = StandardScaler()
    # sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # print(np.sum(y_train == 0))
    # print(np.sum(y_train == 1))
    # print(np.sum(y_test == 0))
    # print(np.sum(y_test == 1))

    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test  = np.c_[X_test,  np.ones(X_test.shape[0])]

    return X_train, X_test, y_train, y_test

def process_credit(percentile=None, n_neg_samples=20000):
    df = pd.read_csv('data/creditcard.csv')
    df.drop(columns=['Time'], axis=0, inplace=True)
    n_neg_samples = int(n_neg_samples)

    if n_neg_samples is not None:
        df_pos = df[df.Class == 1].reset_index()
        df.drop(df[df.Class == 1].index, inplace=True)
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = pd.concat([df.head(n_neg_samples), df_pos], axis=0)
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df.drop(columns=['index'], axis=0, inplace=True)

    X = df.drop(columns=['Class']).to_numpy()
    y = df['Class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("Running preprocessing . . .")
    X_train, X_test, y_train, y_test = process_telco()
    X_train, X_test, y_train, y_test = process_adult()
    X_train, X_test, y_train, y_test = process_credit()
    print("Finished preprocessing.")
