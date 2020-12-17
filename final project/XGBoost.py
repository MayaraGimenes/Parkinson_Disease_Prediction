import inline as inline
import matplotlib
import pandas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import preprocessing
from itertools import combinations

bankdata = pandas.read_csv("C:\\Users\\MayaraPC\\Documents\\UDEL\\SPRING2020\\844\\final project\\dataset\\parkinsons.data")
X = bankdata.drop(['status', 'name'], axis=1)
y = bankdata['status']


print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
cor = pd.concat([X_train,y_train],axis=1).corr()
cor_status = (np.abs(cor['status'].to_numpy())).argsort()
Nfeatures=10
idx_s = cor_status[-2:(Nfeatures+1):-1]

min_max_scaler = preprocessing.MinMaxScaler()
X_train_rangenorm = min_max_scaler.fit_transform(X_train)
X_test_rangenorm = min_max_scaler.transform(X_test)

X_train_c = pd.DataFrame(X_train_rangenorm, columns = X_train.columns)
X_test_c = pd.DataFrame(X_test_rangenorm, columns = X_test.columns)

#DataFlair - Train the model
model=XGBClassifier()
model.fit(X_train,y_train)


# DataFlair - Calculate the accuracy
y_pred=model.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)