import csv
import random
import math
import operator
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()

x=iris.data
y=iris.target

x_train,x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)

class MaxMinscaler():
    import numpy as np
    def fit(self,x):
        self.min_=x.min(axis=0)
        self.max_=x.max(axis=0)
        self.range_=self.max_-self.min_
        
    def transform(self,x):
        self.min_1=np.tile(self.min_,(x.shape[0],1))
 
        self.range_1=np.tile(self.range_,(x.shape[0],1))

        return (x-self.min_1)/self.range_1       

def predict(xtest, ytest, xtrain, ytrain, k=5):
    
    # max_min scale the x_train and x_test
    maxmin=MaxMinscaler()
    maxmin.fit(xtrain)
    x_train_sc=maxmin.transform(xtrain)
    x_test_sc=maxmin.transform(xtest)
    
    class_vote=[]
    for i in range(xtest.shape[0]):
        diff=x_train_sc-x_test_sc[i]
        distances=np.linalg.norm(diff,axis=1)
        idx=np.argsort(distances)[:k]
        classes=y_train[idx]
        count={}
        for j in range(len(classes)):
            if classes[j] in count:
                count[classes[j]]+=1
            else:
                count[classes[j]]=1
        class_vote.append(sorted(count.items(), key=operator.itemgetter(1), reverse=True)[0][0])
    return np.array(class_vote)

res=predict(x_test,y_test,x_train,y_train)


# use KNeighborsClassifier in sklearn to verify our self-defined function
knn=neighbors.KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.predict(x_test)
