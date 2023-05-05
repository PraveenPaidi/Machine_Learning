#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import r2_score

## Read train CSV
df = pd.read_csv(r'D:\Spring 23\EEE 591\regressionprob1_train0.csv')
##read test data csv
dftest = pd.read_csv(r'D:\Spring 23\EEE 591\regressionprob1_test0.csv')

# DATA formualtion for train and test
x = df.iloc[:,0:4].values
y = df['F'].values
xtest = dftest.iloc[:,0:4].values
ytest = dftest['F'].values
Train_Size = len(y)
Test_Size = len(ytest)

# creating compact notation
Ones = np.ones(Train_Size,dtype=float).reshape(Train_Size,1)  
X_train = np.hstack((Ones,x))
# weights 
W = np.matmul(np.linalg.inv(np.dot(X_train.T,X_train)),np.dot(X_train.T,y))
# Prediciting the values for linalginv
Y_atrain= np.matmul(X_train,W)      # err=np.matmul(X,w)
R2a =r2_score(y,Y_atrain)     # r=Rsquared(y,err)


#solve using numpy.linalg.solve (b part)
W_ = np.linalg.solve(np.dot(X_train.T,X_train),np.dot(X_train.T,y))
# Prediciting the values for linalgsolve
Y_btrain= np.matmul(X_train,W_)
R2b = r2_score(y,Y_btrain)


# C part TESTING THE MODEL
Ones = np.ones(Test_Size,dtype=float).reshape(Test_Size,1)  ## make a column vector of ones
# creating compact notation
X_test = np.hstack((Ones,xtest))
#linalginv solution
Y_atest = np.matmul(X_test,W)
R2atest=r2_score(ytest,Y_atest)
#linalgsolve solution
Y_btest= np.matmul(X_test,W_)
R2btest=r2_score(ytest,Y_btest)

print('Q4.a) Intercept : ', W[0], '\nweights: ', W[1:], '\nResidual Squared : ', R2a)
print('\nQ4.b) Intercept : ', W_[0], '\nweights: ', W_[1:], '\nResidual Squared : ', R2b)
print('Q4.c) \n Residual Squared from linalginv is : ', R2atest, '\n Residual Squared  from linalgsolve is : ', R2btest)


# QUESTION 5 GRADIENT DESCENT
error = 0.0000000000001
alpha = 0.001 
COUNT= 1
Converged=False
## Initiate Weights
W = np.ones(5)

while(not Converged): 
    Wnew = W - alpha*(2/Train_Size)*np.dot((np.dot(X_train,W)-y),X_train)
    err = np.dot(Wnew - W,Wnew - W) 
    COUNT += 1
    W = Wnew
    Converged=False

    if np.abs(err)<error :
        Converged=True
# Predicting Y
ypred= np.dot(X_test,W)
r2gradtest=r2_score(ytest,ypred)
print('\nQ5) Intercept : ', W[0], '\nweights: ', W[1:], '\nResidual Squared : ', r2gradtest)


#Question 6
##sklearn.linear_model.LinearRegression
slr = lr().fit(X_train, y)
ylrtest = slr.predict(X_test)
r2=slr.score(X_test,ytest)
print ('\nQ6) intercept :', slr.intercept_ ,'\nweights:', slr.coef_[1:], '\nResidual Squared : ', r2)

