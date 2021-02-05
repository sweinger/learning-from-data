#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:14:53 2020

@author: sweinger
"""

import matplotlib.pyplot as plt
import numpy as np
import random

# Error Hoeffding inequality

nu_1s = np.zeros(100000)
nu_rands = np.zeros(100000)
nu_mins = np.zeros(100000)

for t in range(0, 100000):
    
    flips = [[random.randint(0, 1) for x in range(0, 10)] for y in range(0, 1000)]
    nu_1s[t] = sum(flips[0])/10
    nu_rands[t] = sum(random.choice(flips))/10
    nu_mins[t] = min([sum(x) for x in flips])/10
    

# Linear regression
# change N to choose number of points to train PLA
N = 100   
    
#    plt.plot(x, y, '-r', label='f')
#    plt.title('Graph of f')
#    plt.grid()
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)
#    plt.show() 

fs = []
gs = []
E_ins = []

for t in range(0, 1000):   
    
    f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
    
    x = np.linspace(-1, 1, 100)
    m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
    b = f[0][1] - m*f[0][0]
    y = m*x+b     
    
#    Use Linear Regression to find g and evaluate Ein, the fraction of in-sample points which got classified incorrectly
        
    X = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y = [-1 if x_i[:,2] < (m*x_i[:,1]+b) else 1 for x_i in X]
    w = (X.transpose().dot(X)).I.dot(X.transpose()).dot(Y)
    
#    reds = [X[i] for i in range(0, N) if Y[i] == -1]
#    greens= [X[i] for i in range(0, N) if Y[i] == 1]
#    
#    plt.plot(x, y, '-r', label='f')
#    plt.scatter(x=[r[:,1] for r in reds], y=[r[:,2] for r in reds], color='red')
#    plt.scatter(x=[g[:,1] for g in greens], y=[g[:,2] for g in greens], color='green')
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)    
#    plt.show()
    y_hat = np.sign(w.dot(X.transpose()))
    E_in = len([i for i in range(0, N) if y_hat[0,i] != Y[i]])/N
    
    fs.append((m, b))
    gs.append(w)
    E_ins.append(E_in)
    
#  generate 1000 fresh points and use them to estimate the out-of-sample error Eout of g

N_out = 1000 
E_outs = []  
    
for t in range(0, 1000):
    
    m = fs[t][0]
    b = fs[t][1]    
    
    X_out = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N_out)])
    Y_out = [-1 if x_i[:,2] < (m*x_i[:,1]+b) else 1 for x_i in X_out]

    y_hat = np.sign(gs[t].dot(X_out.transpose()))
    E_out = len([i for i in range(0, N_out) if y_hat[0,i] != Y_out[i]])/N_out #np.mean([calculate_E_out(g, X_out, Y_out) for g in gs])
    E_outs.append(E_out)    
        
    
N = 10

iterations = []

for t in range(0, 1000):
    
    f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
    
    x = np.linspace(-1, 1, 100)
    m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
    b = f[0][1] - m*f[0][0]
    y = m*x+b  

    X = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y = [-1 if x_i[:,2] < (m*x_i[:,1]+b) else 1 for x_i in X]
    w = (X.transpose().dot(X)).I.dot(X.transpose()).dot(Y) 
    
#    reds = [X[i] for i in range(0, N) if Y[i] == -1]
#    greens= [X[i] for i in range(0, N) if Y[i] == 1]
#    
#    plt.plot(x, y, '-r', label='f')
#    plt.scatter(x=[r[:,1] for r in reds], y=[r[:,2] for r in reds], color='red')
#    plt.scatter(x=[g[:,1] for g in greens], y=[g[:,2] for g in greens], color='green')
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)    
#    plt.show()    

    # and at each iteration have the algorithm choose a point randomly from the set of misclassified points.
    
    for s in range(0, 10000):
        H = np.sign(w.dot(X.transpose())) #[int(np.sign(w[0]+w[1]*w[i][0]+W[2]*X[i][1])) for i in range(0, N)]
        misses = [i for i in range(0, N) if H[:,i] != Y[i]]
        if len(misses) == 0:
            break;
        p = random.choice(misses)
        w[:,0] = w[:,0] + Y[p]*1
        w[:,1] = w[:,1] + Y[p]*X[p,1]
        w[:,2] = w[:,2] + Y[p]*X[p,2]     

    iterations.append(s)
    
print(np.mean(iterations))

N = 1000

E_ins = []

# Carry out Linear Regression without transformation

for t in range(0, 1000):

    X = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y = np.sign(np.asarray(X[:,1])**2 + np.asarray(X[:,2])**2 - 0.6).flatten()
    noise = np.random.choice(N, np.int(N*0.1))
    Y[noise] = Y[noise]*-1
    
    w = (X.transpose().dot(X)).I.dot(X.transpose()).dot(Y)
    y_hat = np.sign(w.dot(X.transpose()))
    E_in = len([i for i in range(0, N) if y_hat[0,i] != Y[i]])/N
    E_ins.append(E_in)
    
print(np.mean(E_ins))

N = 1000

E_ins = []

# transform the N = 1000 training data into the following nonlinear feature vector:
# (1,x1,x2,x1*x2,x1**2,x2**2)

ws = []

for t in range(0, 100):

    X = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y = np.sign(np.asarray(X[:,1])**2 + np.asarray(X[:,2])**2 - 0.6).flatten()
    noise = np.random.choice(N, np.int(N*0.1))
    Y[noise] = Y[noise]*-1
    
    X_transform = np.concatenate([X, np.asarray(X[:,1])*np.asarray(X[:,2]), np.asarray(X[:,1])**2, np.asarray(X[:,2])**2], axis=1)
    
    w = (X_transform.transpose().dot(X_transform)).I.dot(X_transform.transpose()).dot(Y)
    ws.append(w)
#    y_hat = np.sign(w.dot(X_transform.transpose()))
#    E_in = len([i for i in range(0, N) if y_hat[0,i] != Y[i]])/N
#    E_ins.append(E_in)
    
w = np.matrix([
        np.mean([w[:,0] for w in ws]),
        np.mean([w[:,1] for w in ws]),
        np.mean([w[:,2] for w in ws]),
        np.mean([w[:,3] for w in ws]),
        np.mean([w[:,4] for w in ws]),
        np.mean([w[:,5] for w in ws])
        ])

N_out = 1000 
E_outs = []

for t in range(0, 1000):
    X_out = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N_out)])
    Y_out = np.sign(np.asarray(X_out[:,1])**2 + np.asarray(X_out[:,2])**2 - 0.6).flatten()
    noise = np.random.choice(N_out, np.int(N_out*0.1))
    Y_out[noise] = Y_out[noise]*-1
    
    X_out_transform = np.concatenate([X_out, np.asarray(X_out[:,1])*np.asarray(X_out[:,2]), np.asarray(X_out[:,1])**2, np.asarray(X_out[:,2])**2], axis=1)
    
    y_hat = np.sign(w.dot(X_out_transform.transpose()))
    E_out = len([i for i in range(0, N_out) if y_hat[0,i] != Y_out[i]])/N_out
    E_outs.append(E_out)      
    
print(np.mean(E_outs))    