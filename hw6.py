#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:56:15 2020

@author: sweinger
"""


train = np.loadtxt('/Users/sweinger/Documents/caltech_course/in.dta.txt')
test = np.loadtxt('/Users/sweinger/Documents/caltech_course/out.dta.txt')
X = train[:,0:2]
Y = train[:,2]
X_out = test[:,0:2]
Y_out = test[:,2]

N = X.shape[0]
N_out = X_out.shape[0]

reds = [X[i] for i in range(0, N) if Y[i] == -1]
greens= [X[i] for i in range(0, N) if Y[i] == 1]

# for visualization purposes
plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
plt.xlim(-1, 1)
plt.ylim(-1, 1)    
plt.show()  

Z = np.matrix(np.vstack(([1]*N,X[:,0],X[:,1],X[:,0]**2,X[:,1]**2,X[:,0]*X[:,1],np.abs((X[:,0]-X[:,1])),np.abs((X[:,0]+X[:,1])))).T)
Z_out = np.matrix(np.vstack(([1]*N_out,X_out[:,0],X_out[:,1],X_out[:,0]**2,X_out[:,1]**2,X_out[:,0]*X_out[:,1],np.abs((X_out[:,0]-X_out[:,1])),np.abs((X_out[:,0]+X_out[:,1])))).T)

def classification_errors(k=None):
    penalty = 10**k if k is not None else 0
    w = (Z.transpose().dot(Z)+penalty*np.identity(Z.shape[1])).I.dot(Z.transpose()).dot(Y)
    y_hat = np.sign(w.dot(Z.transpose()))
    Ein = 1-np.mean(y_hat == Y)    
    y_hat_out = np.sign(w.dot(Z_out.transpose()))
    Eout = 1-np.mean(y_hat_out == Y_out)    
    return([Ein, Eout])    

# linear regression
display(classification_errors())

# with regularization k = -3
display(classification_errors(k=-3))

# with regularization k = 3
display(classification_errors(k=3))

candidate_ks = [2, 1, 0, -1, -2]
oos_errors = []
for k in candidate_ks:
    oos_errors.append(classification_errors(k=k)[1])

display(candidate_ks[np.argmin(oos_errors)])
display(min(oos_errors))
