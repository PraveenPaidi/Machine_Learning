#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,0:4]
Y = iris.target

# Compact Notation
m = np.size(Y)
Ones = np.ones(m,dtype=int) 
X_compact = np.column_stack((Ones,X)) 

X_train,X_test,Y_train,Y_test = train_test_split(X_compact,Y,test_size=0.3,random_state=0)

# One hot Encoding
targets = np.array(Y_train).reshape(-1)
Y_train_one= np.eye(3)[targets]

targets = np.array(Y_test).reshape(-1)
Y_test_one= np.eye(3)[targets]

converged=False
W = np.ones((5,3))
alpha=0.1
iterations=0
Training_error=[]
Testing_error=[]

def sigmoid(X,W):
    return 1/(1 + np.exp(-(np.dot(X.T,W))))

while not converged:   
    U = - 1/m * np.dot(X_train.T,(Y_train_one - sigmoid(X_train.T,W)))
    W_new = W - alpha * U
    epsilon=np.abs(W_new-W)
    W = np.copy(W_new)
     
    converged=(np.abs(epsilon) < 0.0001).all()
    if iterations>4000:
        converged = True
    
    Y_train_pred=np.dot(X_train,W)
    Y_train_pred1=np.argmax(Y_train_pred,axis=1)
    Train_error = np.sum((Y_train != Y_train_pred1)/np.size(Y_train))
    
    Y_test_pred=np.dot(X_test,W)
    Y_test_pred1=np.argmax(Y_test_pred,axis=1)
    Test_error = np.sum((Y_test != Y_test_pred1)/np.size(Y_test))
    
    Training_error.append(Train_error)
    Testing_error.append(Test_error)
    
    iterations+=1
print('Question 4')
print('Multiclass')
print('Learning rate is 0.1')
print('Train_errors are ', Train_error*np.size(Y_train))
print('Test_errors are ', Test_error*np.size(Y_test))
print("Testing accuracy =",(np.size(Y_test) - Test_error*np.size(Y_test))/np.size(Y_test))
print("Training accuracy =",(np.size(Y_train) - Train_error*np.size(Y_train))/np.size(Y_train))

def mosthighlycorrelated(mydataframe, numtoreport):  
    cormatrix = mydataframe.corr() 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    return cormatrix.head(numtoreport)

irisp = pd.DataFrame(iris.data,columns=iris.feature_names) 

print("\nMost Highly Correlated") 
print(mosthighlycorrelated(irisp,50)) 
print('\n',irisp.head())

print('\n Question 5')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,solver='lbfgs', max_iter=2000).fit(X_train,Y_train)
print("Canned algorithm Logistic Regression")
print("Training Accuracy is ",clf.score(X_train, Y_train))
print("Testing Accuracy is ",clf.score(X_test, Y_test))

