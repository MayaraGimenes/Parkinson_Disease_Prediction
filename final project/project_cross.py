import inline as inline
import matplotlib
import pandas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from itertools import combinations



bankdata = pandas.read_csv("C:\\Users\\MayaraPC\\Documents\\UDEL\\SPRING2020\\844\\final project\\dataset\\parkinsons.data")
X = bankdata.drop(['status', 'name'], axis=1)
y = bankdata['status']


print(X.shape)
cor = pd.concat([X,y],axis=1).corr()
cor_status = (np.abs(cor['status'].to_numpy())).argsort()
Nfeatures=10
idx_s = cor_status[-2:(Nfeatures+1):-1]


#Neural network
v=0
list_acc = []
# for i in range(1, len(idx_s)+1):
#     print(i)
#     comb = combinations(idx_s, i)
#     for k in list(comb):
v=v+1
#X_t = X#X.iloc[:,list(k)] #original dataset
X_t = X.iloc[:,idx_s] #dataset with the most correlated

svclassifier = MLPClassifier(solver='adam',activation='relu', alpha=1e-4, hidden_layer_sizes = (40, 20, 10), random_state = 1, max_iter=2000)
accuracy = cross_val_score(svclassifier, X_t, y, scoring='accuracy', cv=10)
print("Accuracy of Model ", v ," with Cross Validation is:", accuracy.mean() * 100)
list_acc.append(accuracy.mean() * 100)

print(v)
print(list_acc)
print(max(list_acc))
exit()
#SVM
v=0
list_acc = []
for i in range(1, len(idx_s)+1):
    print(i)
    comb = combinations(idx_s, i)
    for k in list(comb):
        v=v+1
        X_t = X.iloc[:,list(k)]
        svclassifier = SVC(kernel='rbf')
        accuracy = cross_val_score(svclassifier, X_t, y, scoring='accuracy', cv=10)
        print("Accuracy of Model ", v ," with Cross Validation is:", accuracy.mean() * 100)
        list_acc.append(accuracy.mean() * 100)

print(v)
print(list_acc)
print(max(list_acc))