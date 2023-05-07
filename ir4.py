import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv('iris.csv')

X=np.array(d)[:,:-1]
y=np.array(d)[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

k_range= range (1,26)
scores=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,scores)
plt.title('graphical representation')
plt.show()
