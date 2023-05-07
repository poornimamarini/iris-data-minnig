import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv('iris.csv')

X=np.array(d)[:,:-2]
y=np.array(d)[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
print(type(X))
print('data')
print(X.shape)
print(X[:40])
print(X[41:80])
print(X[81:120])
print(X[121:150])
print('target')
print(y)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print('X_train')
print(X_train.shape)
print(X_train[:40])
print(X_train[41:75])

print('X_test')
print(X_test.shape)
print(X_test[:40])
print(X_test[41:75])
