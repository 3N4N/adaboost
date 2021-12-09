import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

iris = pd.read_csv('data/Iris.csv')

iris.head()
iris.info()


data = iris.drop('Id', axis=1)
X = data.iloc[: , :-1]
y = data.iloc[: , -1]
print("Shape of X is %s and shape of y is %s"%(X.shape,y.shape))

total_classes = y.nunique()
print("Number of unique species in dataset are: ",total_classes)

distribution = y.value_counts()
print(distribution)

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.25, random_state=28)

# Creating adaboost classifier model
adb = AdaBoostClassifier()
adb_model = adb.fit(X_train, Y_train)

print("The accuracy of the model on validation set is", adb_model.score(X_val,Y_val))
