#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:41:19 2020

@author: sweinger
"""

w = [1,1]

nu = 0.1
epsilon = 10E-14
t = 0

# need to calculate:
# Ein(u,v)
# partial of Ein w.r.t u
# partial of Ein w.r.t v

def E(u, v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2

def dE_du(u, v):
    return 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))

def dE_dv(u, v):
    return 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))

while True:
    
#    w[0] = w[0] - nu*dE_du(w[0], w[1])
#    w[1] = w[1] - nu*dE_dv(w[0], w[1])
    w = [w[0] - nu*dE_du(w[0], w[1]), w[1] - nu*dE_dv(w[0], w[1])]
    if E(w[0], w[1]) < epsilon:
        break
    t = t + 1    


# coordinate descent

w = [1,1]

nu = 0.1
epsilon = 10E-14
    
for t in range(0, 15):
    
    w[0] = w[0] - nu*dE_du(w[0], w[1])
    w[1] = w[1] - nu*dE_dv(w[0], w[1])
    
print(E(w[0], w[1]))


# logistic regression
N = 100
nu = 0.01

# run stochastic gradient descent on X to find g

def dE_dg(x, y, g):
    return -(y*x)/(1+np.exp(y*g.dot(x.transpose())))

def E_in(X, Y, g):
    s = 0
    for i in range(0, X.shape[0]):
        s = s + np.log(1+np.exp(-Y[i]*g.dot(X[i,:].transpose())[0,0]))
    return s/X.shape[0]

B = 100

e_outs = []
ts = []

for r in range(0, B):
    
    f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
    m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
    b = f[0][1] - m*f[0][0]
        
    X = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y = [-1 if x_i[:,2] < (m*x_i[:,1]+b) else 1 for x_i in X]
    
#    reds = [X[i] for i in range(0, N) if Y[i] == -1]
#    greens= [X[i] for i in range(0, N) if Y[i] == 1]
#    
#    # for visualization purposes
#    x = np.linspace(-1, 1, 100)
#    y = m*x+b 
#    
#    plt.plot(x, y, '-r', label='f')
#    plt.scatter(x=[r[:,1] for r in reds], y=[r[:,2] for r in reds], color='red')
#    plt.scatter(x=[g[:,1] for g in greens], y=[g[:,2] for g in greens], color='green')
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)    
#    plt.show()  
    
# Initialize the weight vector of Logistic Regression to all zeros in each run. 
# Stop the algorithm when ||w(t−1) − w(t)|| < 0.01
# where w(t) denotes the weight vector at the end of epoch t.    
    
    
    g = np.array([0, 0, 0])

    t = 0

# epochs
    
    while True:
                
        indices = np.random.choice(N, size=N, replace=False)

        g_t = g
        
        for i in indices:        
            x = X[i,:]
            y = Y[i]
            g_t = g_t - nu*dE_dg(x, y, g_t)

        norm = np.linalg.norm(g_t-g)
        g = g_t
        t = t + 1
        if norm < 0.01:
            break
    
    X_out = np.matrix([(1, random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    Y_out = [-1 if x_i[:,2] < (m*x_i[:,1]+b) else 1 for x_i in X_out]
    
    E_out = E_in(X_out, Y_out, g)
    e_outs.append(E_out)
    ts.append(t)
    
print(np.mean(e_outs))    
print(np.mean(ts))    