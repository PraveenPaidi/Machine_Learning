#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

X = np.array([[-2,4,-1],[4,1,-1],[1,6,-1],[2,4,-1],[6,2,-1]]) 
Y = np.array([-1,-1,1,1,1])  
alpha = 0.1
errors = []
converged= False
count=0
misclassification=0
S=len(X[:,0])
ones= np.ones(S,dtype=float).reshape(S,1)
A=np.hstack((ones, X))
w = np.random.rand(len(A[0]))

print('Q2')
while not converged:
    count+=1
    total_error = 0.0 
    
    for i in range(len(Y)):
        error = -np.dot(A[i], w) * Y[i]
        if (error >= 0):
            w = w + alpha*A[i]*Y[i]
            total_error += error       
    ypred = np.dot(A,w)
    errors.append(total_error)
    
    for i in range (len(ypred)):
        if ypred[i]<0:
            ypred[i]=-1
        else:
            ypred[i]=1
 
    if count>15:
        converged=True
    print( '\nIteration',count,'\n weights ', w, '\n error ', error, '\nprediction' ,ypred) 

plt.plot(errors)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Iterations vs error')
plt.show()
print('Accuracy of model: %.2f' % accuracy_score(Y, ypred))
for i in range(len(Y)):
    if ypred[i]!=Y[i]:
        misclassification+=1
print('Number of misclassification predictions are' ,misclassification)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_adaline = pd.read_csv(r'D:\Spring 23\EEE 591\HW2\Dataset_1.csv')
X_train1 = df_adaline.iloc[:,0:3].values
Y_train1 = df_adaline['y'].values
stdslr = StandardScaler()
X_trainstd1 = stdslr.fit_transform(X_train1[:,0:3])
one = np.ones(len(X_trainstd1[:,0]),dtype=float)
# compact notation
A_a1 = np.column_stack((one,X_trainstd1))

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(A_a1,Y_train1,test_size=0.33,random_state=0)
no_of_obs = len(Y_train1)

def adaline(X, w, Y, eta):
    errors = []
    total_error = 1.0
    last_total_error = 0
    A = X
    count=0
    converged= False
    while not converged:
        y_pred = np.dot(A,w)
        Error = y_pred - Y
        w = w - eta *(2/no_of_obs)*np.dot(A.T,Error)
        total_error = np.dot(Error,Error) 
        errors.append(total_error/len(Y))
        count += 1
        if count>50 or abs(total_error - last_total_error) < 0.001:
            converged=True
        last_total_error = total_error
    return w,errors,count  

w = np.zeros(len(X_train1[0]))    
w,errors,t = adaline(X_train1,w,Y_train1,0.1) 

print('Q4')
plt.plot(errors)
plt.title('Error vs Iteration')
plt.xlabel('iterations')
plt.ylabel('errors')
plt.show()
print('Q4. DATASET1')
print(' Weights ', w)
print(' No. of Iterations:', t)
def squared_loss(X,w,y):
    Sq_error = 0.0
    misclassified=0
    ypred = np.dot(X,w)
    for i in range(len(y)):
        if (ypred[i] >= 0.0):
            ypred[i]=1
        else:
            ypred[i]=-1
        error = y[i]-ypred[i]
        Sq_error += error*error
        if y[i]!=ypred[i]:
            misclassified+=1
     
    print(' Misclassified observations:', misclassified)
    print(' Total Error ', Sq_error)
    print(' Accuracy:  %.2f' % accuracy_score(y, ypred))
    return ypred,Sq_error

print('\n For training data')
ypred_train,sq_error = squared_loss(X_train1,w,Y_train1)
print('\n For test data')
ypred_test,sq_error = squared_loss(X_test1,w,Y_test1)

print('Q4. DATASET2')
df_adaline2 = pd.read_csv(r'D:\Spring 23\EEE 591\HW2\Dataset_2.csv')
X_train2 = df_adaline2.iloc[:,0:3].values
Y_train2 = df_adaline2['y'].values
X_trainstd2 = stdslr.fit_transform(X_train2[:,0:3])
one = np.ones(len(X_trainstd2[:,0]),dtype=float)
# compact notation
A_a2 = np.column_stack((one,X_trainstd2))

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(A_a2,Y_train2,test_size=0.33,random_state=0)
w_init = np.ones(len(X_train2[0]))*0.1
w_2,errors_tot,t = adaline(X_train2,w_init,Y_train2,0.1) 

print('\n For training data')
yp_train,total_error = squared_loss(X_train2,w_2,Y_train2)
print('\n For test data')
yp_test,tot_err = squared_loss(X_test2,w_2,Y_test2)

print(' FROM SKLEARN')
print('\nQ5 Data set 1')


from sklearn.linear_model import Perceptron
P = Perceptron(max_iter=1500, tol=0.000001, eta0=0.001, random_state=0)
P.fit(X_train1, Y_train1)
y_pred = P.predict(X_train1)
print(' \n Training Data'' \nMisclassified samples: %d' % (Y_train1 != y_pred).sum())
print(' Train Accuracy: %.2f' % accuracy_score(Y_train1, y_pred))

y_pred = P.predict(X_test1)
print(' \n Testing data ''\nMisclassified samples: %d' % (Y_test1 != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(Y_test1, y_pred))


print('\n Data set 2')
P2 = Perceptron(max_iter=1500, tol=0.0000001, eta0=0.001, random_state=0)
P2.fit(X_train2, Y_train2)
y_pred2 = P2.predict(X_train2)
print(' \n Training data ''\nMisclassified samples: %d' % (Y_train2 != y_pred2).sum())
print('Train Accuracy: %.2f' % accuracy_score(Y_train2, y_pred2))

y_pred2 = P2.predict(X_test2)
print('\n testing data''\nMisclassified samples: %d' % (Y_test2 != y_pred2).sum())
print('Test Accuracy: %.2f' % accuracy_score(Y_test2, y_pred2))


# In[ ]:




