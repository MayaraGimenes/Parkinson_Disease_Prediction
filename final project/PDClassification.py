import inline as inline
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import combinations
from xgboost import XGBClassifier


def xgboost(idx_s,X,y):
    v=0
    list_acc = []
    for i in range(1, len(idx_s)+1):
        print(i)
        comb = combinations(idx_s, i)
        for k in list(comb):
            v=v+1
            nam=str()
            for dd in range(len(list(k))):
                nam= nam+(X.iloc[:,k[dd]].name + " ")
            X_t = X.iloc[:,list(k)]
            XGBoostModel = XGBClassifier(objective="binary:logistic", random_state=42)
            accuracy = cross_val_score(XGBoostModel, X_t, y, scoring='accuracy', cv=10)
            print('Features ' + nam)
            print("Accuracy of XGBoost Model ", v ," with Cross Validation is:", accuracy.mean() * 100)
            list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.plot(range(1,v+1),list_acc,)
    plt.title('XGBoost accuracy vs Selected Features Index')
    plt.xlabel('Selected Features Index')
    plt.ylabel('Accuracy')
    plt.savefig("XGBoost.png")
    plt.close()
#RF
def randomforest(X,y):
    list_acc = []
    X_t = X
    estim = range(20,800,10)
    for n_est in estim:
        RFmodel = RandomForestClassifier(n_estimators=n_est, random_state=0)
        accuracy = cross_val_score(RFmodel, X_t, y, scoring='accuracy', cv=10)
        print("Accuracy of Random Forest Model  with Cross Validation is:", accuracy.mean() * 100)
        list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.plot(estim,list_acc,'x')
    plt.title('Random Forest accuracy vs Number of Estimators')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.savefig("RF.png")
    plt.close()

#NN
def NeuralNetwork(X,y):
    list_acc = []

    X_t = X
    hn = range(5,50,5)
    for i in hn:
        for k in hn:
            NN = MLPClassifier(solver='adam', activation='relu', alpha=1e-4, hidden_layer_sizes=(i, k), random_state=1, max_iter=5000)
            accuracy = cross_val_score(NN, X_t, y, scoring='accuracy', cv=10)
            print("Accuracy of Relu Neural Network Model with Cross Validation is:", accuracy.mean() * 100 , " HN-L1=", i , " HN-L2=",k)
            list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.imshow(np.reshape(list_acc, (len(hn),len(hn))), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(hn)), hn)
    plt.yticks(range(len(hn)), hn)
    plt.ylabel('Hidden Nodes - Layer 1')
    plt.xlabel('Hidden Nodes - Layer 2')
    plt.title('Relu Neural Network accuracy with different number of Hidden Nodes')
    plt.savefig("ReluNN.png")
    plt.close()
    list_acc = []

    for i in hn:
        for k in hn:
            NN = MLPClassifier(solver='adam', activation='tanh', alpha=1e-4, hidden_layer_sizes=(i, k), random_state=1, max_iter=2000)
            accuracy = cross_val_score(NN, X_t, y, scoring='accuracy', cv=10)
            print("Accuracy of Tanh Neural Network Model with Cross Validation is:", accuracy.mean() * 100 , " HN-L1=", i , " HN-L2=",k)
            list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.imshow(np.reshape(list_acc, (len(hn),len(hn))), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(hn)), hn)
    plt.yticks(range(len(hn)), hn)
    plt.ylabel('Hidden Nodes - Layer 1')
    plt.xlabel('Hidden Nodes - Layer 2')
    plt.title('Tanh Neural Network accuracy with different number of Hidden Nodes')
    plt.savefig("TanhNN.png")
    plt.close()
    list_acc = []

    for i in hn:
        for k in hn:
            NN = MLPClassifier(solver='adam', activation='logistic', alpha=1e-4, hidden_layer_sizes=(i, k), random_state=1, max_iter=5000)
            accuracy = cross_val_score(NN, X_t, y, scoring='accuracy', cv=10)
            print("Accuracy of Logistic Neural Network Model with Cross Validation is:", accuracy.mean() * 100 , " HN-L1=", i , " HN-L2=",k)
            list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.imshow(np.reshape(list_acc, (len(hn),len(hn))), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(hn)), hn)
    plt.yticks(range(len(hn)), hn)
    plt.ylabel('Hidden Nodes - Layer 1')
    plt.xlabel('Hidden Nodes - Layer 2')
    plt.title('Logistic Neural Network accuracy with different number of Hidden Nodes')
    plt.savefig("LogiNN.png")
    plt.close()# def NeuralNetwork(idx_s,X,y):

#SVM
def SVM(idx_s,X,y):
    v=0
    list_acc = []
    for i in range(1, len(idx_s)+1):
        print(i)
        comb = combinations(idx_s, i)
        for k in list(comb):
            v=v+1
            X_t = X.iloc[:,list(k)]
            svclassifier = SVC(kernel='rbf')
            nam=str()
            for dd in range(len(list(k))):
                nam= nam+(X.iloc[:,k[dd]].name + " ")
            accuracy = cross_val_score(svclassifier, X_t, y, scoring='accuracy', cv=10)
            print('Features ' + nam)
            print("Accuracy of SVM Model ", v ," with Cross Validation is:", accuracy.mean() * 100)
            list_acc.append(accuracy.mean() * 100)
    plt.figure()
    plt.plot(range(1,v+1),list_acc,)
    plt.title('SVM accuracy vs Selected Features Index')
    plt.xlabel('Selected Features Index')
    plt.ylabel('Accuracy')
    plt.savefig("SVM.png")
    plt.close()

bankdata = pandas.read_csv("./final project/dataset/parkinsons.data")
X = bankdata.drop(['status', 'name'], axis=1)
y = bankdata['status']

print(X.shape)
cor = pd.concat([X,y],axis=1).corr()
cor_status = (np.abs(cor['status'].to_numpy())).argsort()
Nfeatures=10
idx_s = cor_status[-2:(-Nfeatures-2):-1]

for i in range(X.shape[1]):
    plt.figure()
    plt.plot(X.iloc[:,i].to_numpy(),y.to_numpy(),'x')
    plt.xlabel(X.iloc[:,i].name)
    plt.ylabel(y.name)
    if i in idx_s:
        plt.title(X.iloc[:,i].name + " vs status - Chosen - Correlation=" +str(cor['status'].iloc[i]) )
    else:
        plt.title(X.iloc[:,i].name + " vs status - Not Chosen - Correlation=" +str(cor['status'].iloc[i]) )

    plt.savefig(X.iloc[:,i].name+".png")
    plt.close()

NeuralNetwork(X,y)
xgboost(idx_s,X,y)
SVM(idx_s,X,y)
randomforest(X,y)

