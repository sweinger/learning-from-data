#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:23:54 2020

@author: sweinger
"""
import matplotlib.pyplot as plt
import numpy as np
import random

# change N to choose number of points to train PLA
N = 100

iterations = []
disagreements = []

for s in range(0, 1000):

    f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
    
    x = np.linspace(-1, 1, 100)
    m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
    b = f[0][1] - m*f[0][0]
    y = m*x+b
    
#    plt.plot(x, y, '-r', label='f')
#    plt.title('Graph of f')
#    plt.grid()
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)
#    plt.show()
    
    X = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)]
    Y = [-1 if x_i[1] < (m*x_i[0]+b) else 1 for x_i in X]
    
#    reds = [X[i] for i in range(0, N) if Y[i] == -1]
#    greens= [X[i] for i in range(0, N) if Y[i] == 1]
#    
#    plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
#    plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
#    plt.show()
    
    W = [0, 0, 0]
    
    # and at each iteration have the algorithm choose a point randomly from the set of misclassified points.
    
    for t in range(0, 10000):
        H = [int(np.sign(W[0]+W[1]*X[i][0]+W[2]*X[i][1])) for i in range(0, N)]
        misses = [i for i in range(0, N) if H[i] != Y[i]]
        if len(misses) == 0:
            break;
        p = random.choice(misses)
        W[0] = W[0] + Y[p]*1
        W[1] = W[1] + Y[p]*X[p][0]
        W[2] = W[2] + Y[p]*X[p][1]
        
    N_out = 10000
    X_out = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N_out)]
    Y_out = [-1 if x_i[1] < (m*x_i[0]+b) else 1 for x_i in X_out]
        
    H_out = [int(np.sign(W[0]+W[1]*X_out[i][0]+W[2]*X_out[i][1])) for i in range(0, N_out)]
    misses = [i for i in range(0, N_out) if H_out[i] != Y_out[i]]    
    disagreement = len(misses)/N_out
    
    iterations.append(t)
    disagreements.append(disagreement)
    
print(np.mean(iterations))
print(np.mean(disagreements))