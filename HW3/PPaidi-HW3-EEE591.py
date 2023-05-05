#!/usr/bin/env python
# coding: utf-8

# In[29]:


from autograd import grad 
from autograd import numpy as np 
from autograd import jacobian
from time import process_time
from matplotlib import pyplot as plt

a = 20
b = np.array([1, -1])
b=b.reshape(2,1)
C = np.array([[1, 0],[0, 2]])
lambdaa=0.5
Totallearningrate=[]
Totaldelay=[]
Totalfinalweights=[]

def f(w):
    return(a+np.dot(b.T,w)+np.dot(w.T,np.dot(C,w))) 

def d(w):
    return(b+2*np.dot(C,w))

def reg(w):
    return(b+2*np.dot(C,w) + 2 * lambdaa * w)

def c(g,w):
    gradient = grad(g)
    Dg = gradient(w)
    return(Dg)

def reggrad(g,w):
    gradient = grad(g)
    Dg = gradient(w)+2 * lambdaa * w
    return(Dg)

z = (1+2.236)/2 
def goldendelta(x4,x1,z):
    return((x4-x1)/z)

def goldensearch(g,w,h,x1,x4,accuracy):
    x2 = x4 - goldendelta(x4,x1,z)
    x3 = x1 + goldendelta(x4,x1,z)
    f1 = g(w-x1*h);
    f2 = g(w-x2*h); 
    f3 = g(w-x3*h);
    f4 = g(w-x4*h);
    i = 0
    error = abs(x4-x1)
    while error > accuracy:
        if (f2<f3):
            x4,f4 = x3,f3
            x3,f3 = x2,f2
            x2 = x4 - goldendelta(x4,x1,z)
            f2 = g(w-x2*h)
        else:
            x1,f1 = x2,f2
            x2,f2 = x3,f3
            x3 = x1 + goldendelta(x4,x1,z)
            f3 = g(w-x3*h)
        i += 1
        error = abs(f4-f1)
    return((x1+x4)/2.0,i,error)

def golden(g,w,h,alpha):
    alpha,iter,error = goldensearch(g,w,h,alpha/10.,alpha*10.0,1e-6)
    return alpha



def grad_des(g,w,alpha,iter,q):
    tbeg = process_time()
    count = 1
    absDg = 1.0
    updatew = 1
    errors = []
    tol=0.000000001
    h=1
    w = np.array([-0.3,0.2]).reshape(2,1)
    while ((count<=iter) and (updatew>tol)):
        
        if q==1:                  # Fixed Learning Rate with Analytical Gradient
            Dg = d(w)
        elif q==0:                # Fixed Learning Rate with Autograd
            Dg = c(g,w)
        elif q==2:                # Learning Rate decreases with Iteration by Analytical Gradient
            Dg = d(w)
            alpha=alpha/2
        elif q==3:                # Learning Rate decreases with Iteration by Autograd  
            Dg = c(g,w)
            alpha=alpha/2  
        elif q==12:               # Fixed Learning Rate with Analytical Gradient
            Dg = reg(w)
        elif q==13:               # Fixed Learning Rate with Autograd
            Dg = reggrad(g,w)
            
            
        if q==4:                             # Normalized_methods with Iteration by Analytical Gradient
            Dg = d(w)
            absDg = np.linalg.norm(Dg)
            w = w - alpha*Dg/absDg  
        elif q==5:                           # Normalized_methods with Iteration by Autograd
            Dg = c(g,w)
            absDg = np.linalg.norm(Dg)
            w = w - alpha*Dg/absDg   
        elif q==6:                           # Lipschitz with Iteration by Analytical Gradient
            w = w 
            Dg = d(w)
            absDg = np.linalg.norm(Dg)
            w = w - alpha*Dg/(2*np.linalg.norm(C))
        elif q==7:                           # Lipschitz with Iteration by Autograd
            w = w 
            Dg = c(g,w)
            absDg = np.linalg.norm(Dg)
            w = w - alpha*Dg/(2*np.linalg.norm(C)) 
             
        elif q==8:                           # Steepest Descents with Iteration by Analytical Gradient
            Dg = d(w) 
            absDg = np.linalg.norm(Dg)  
            alpha = golden(g,w,1,alpha)
            w = w - alpha*1
        
        elif q==9:                           # Steepest Descents with Iteration by Autograd
            Dg = c(g,w) 
            absDg = np.linalg.norm(Dg) 
            alpha = golden(g,w,1,alpha)
            w = w - alpha*1
        
        elif q==10:                           # Steepest momentum Descents with Iteration by Analytical Gradient
            Dg = d(w) 
            absDg = np.linalg.norm(Dg)  
            beta=0.5
            h = beta*h + (1-beta)*Dg
            alpha = golden(g,w,h,alpha)
            w = w - alpha*h
        
        elif q==11:                           # Steepest momentum Descents with Iteration by Autograd
            Dg = c(g,w) 
            absDg = np.linalg.norm(Dg) 
            beta=0.5
            h = beta*h + (1-beta)*Dg
            alpha = golden(g,w,h,alpha)
            w = w - alpha*h
        
        else:       
            absDg = np.linalg.norm(Dg)
            w = w - alpha*Dg  
        updatew = absDg*alpha 
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000 
    
    if q==1:                  
        print('Fixed Learning Rate with Analytical Gradient')
    elif q==0:                # Fixed Learning Rate with Autograd
            print('Fixed Learning Rate with Autograd')
    elif q==2:                # Learning Rate decreases with Iteration by Analytical Gradient
            print('Learning Rate decreases with Iteration by Analytical Gradient')
    elif q==3:                # Learning Rate decreases with Iteration by Autograd  
            print('Learning Rate decreases with Iteration by Autograd')
    elif q==4:                             # Normalized_methods with Iteration by Analytical Gradient
            print('Normalized_methods with Iteration by Analytical Gradient')
    elif q==5:                           # Normalized_methods with Iteration by Autograd
            print('Normalized_methods with Iteration by Autograd')
    elif q==6:                           # Lipschitz with Iteration by Analytical Gradient
            print('Lipschitz with Iteration by Analytical Gradient')
    elif q==7:                           # Lipschitz with Iteration by Autograd
            print('Lipschitz with Iteration by Autograd')
    elif q==8:                           # Steepest Descents with Iteration by Analytical Gradient
            print('Steepest Descents with Iteration by Analytical Gradient')
    elif q==9:                           # Steepest Descents with Iteration by Autograd
            print('Steepest Descents with Iteration by Autograd')
    elif q==10:                           # Steepest momentum Descents with Iteration by Analytical Gradient
            print('Steepest momentum Descents with Iteration by Analytical Gradient')
    elif q==11:                           # Steepest momentum Descents with Iteration by Autograd
            print('Steepest momentum Descents with Iteration by Autograd')
    elif q==12:               # Fixed Learning Rate with Analytical Gradient
            print('Fixed Learning Rate with Analytical Gradient with regularizer')
    elif q==13:               # Fixed Learning Rate with Autograd
            print('Fixed Learning Rate with Autograd with regularizer')
    Totallearningrate.append(alpha)
    Totalfinalweights.append(w)
    Totaldelay.append(dtime)
    
                    
    print('Gradient Descents iter ',count)
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('learning rate alpha %.3g ' % alpha)
    print("The final weights:",w)
    print('\n')
    

w = np.array([-0.9,0.6]).reshape(2,1)

dict_classifiers = {

    "Fixed Learning Rate with Analytical Gradient":                  grad_des(f,w,0.1,2000,1),

    "Fixed Learning Rate with Autograd":                             grad_des(f,w,0.1,5000,0),
    
    "Learning Rate decreases with Iteration by Analytical Gradient": grad_des(f,w,0.5,2000,2),

    "Learning Rate decreases with Iteration by Autograd":            grad_des(f,w,0.5,5000,3),
    
    "Normalized_methods with Iteration by Analytical Gradient":      grad_des(f,w,0.5,2000,4),
    
    "Normalized_methods with Iteration by Autograd":                 grad_des(f,w,0.5,5000,5),
    
    "Lipschitz with Iteration by Analytical Gradient":               grad_des(f,w,0.5,2000,6),
    
    "Lipschitz with Iteration by Autograd":                          grad_des(f,w,0.5,2000,7),
    
    "Steepest Descents with Iteration by Analytical Gradient":       grad_des(f,w,0.1,2000,8),
    
    "Steepest Descents with Iteration by Autograd":                  grad_des(f,w,0.1,2000,9),
    
    "Steepest momentum Descents with Iteration by Analytical Gradient":       grad_des(f,w,0.1,2000,10),
    
    "Steepest momentum Descents with Iteration by Autograd":                  grad_des(f,w,0.1,2000,11),
    
    "Fixed Learning Rate with Analytical Gradient":                  grad_des(f,w,0.1,2000,12),

    "Fixed Learning Rate with Autograd":                             grad_des(f,w,0.1,2000,13),
    
}

from tabulate import tabulate
headers = ['dict_classifiers','Totallearningrate','Totaldelay','Totalfinalweights']  
table= zip(dict_classifiers,Totallearningrate,Totaldelay,Totalfinalweights)
print(tabulate(table, headers=headers, floatfmt=".4f"))


# In[ ]:




