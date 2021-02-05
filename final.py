#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:44:38 2020

@author: sweinger
"""

train = np.loadtxt('/Users/sweinger/Documents/caltech_course/features.train.txt')
test = np.loadtxt('/Users/sweinger/Documents/caltech_course/features.test.txt')


X = train[:,1:3]
X_out = test[:,0:2]
Eins = {}

N = X.shape[0]
N_out = X_out.shape[0]


# one vs. all
Z = np.matrix(np.vstack(([1]*N,X[:,0],X[:,1])).T)

penalty = 1
Eins = {}

for digit in [5,6,7,8,9]:

    y = np.where(train[:,0] == digit, 1, -1)
    w = (Z.transpose().dot(Z)+penalty*np.identity(Z.shape[1])).I.dot(Z.transpose()).dot(y)
    y_hat = np.sign(w.dot(Z.transpose()))
    Ein = 1-np.mean(y_hat == y)   
    Eins[digit] = Ein
    
print(min(Eins, key=Eins.get))    

Z = np.matrix(np.vstack(([1]*N,X[:,0],X[:,1],X[:,0]*X[:,1],X[:,0]**2,X[:,1]**2)).T)
Z_out = np.matrix(np.vstack(([1]*N_out,X_out[:,0],X_out[:,1],X_out[:,0]*X_out[:,1],X_out[:,0]**2,X_out[:,1]**2)).T)

Eouts = {}

for digit in [0,1,2,3,4]:

    y = np.where(train[:,0] == digit, 1, -1)
    w = (Z.transpose().dot(Z)+penalty*np.identity(Z.shape[1])).I.dot(Z.transpose()).dot(y)
    y_out = np.where(test[:,0] == digit, 1, -1)    
    y_hat = np.sign(w.dot(Z_out.transpose()))
    Eout = 1-np.mean(y_hat == y_out)
    Eouts[digit] = Eout
    
print(min(Eouts, key=Eouts.get))  


Z = np.matrix(np.vstack(([1]*N,X[:,0],X[:,1])).T)
Z_out = np.matrix(np.vstack(([1]*N_out,X_out[:,0],X_out[:,1])).T)
Z_transform = np.matrix(np.vstack(([1]*N,X[:,0],X[:,1],X[:,0]*X[:,1],X[:,0]**2,X[:,1]**2)).T)
Z_out_transform = np.matrix(np.vstack(([1]*N_out,X_out[:,0],X_out[:,1],X_out[:,0]*X_out[:,1],X_out[:,0]**2,X_out[:,1]**2)).T)

Eins = {}
Eins_transform = {}
Eouts = {}
Eouts_transform = {}

for digit in [0,1,2,3,4,5,6,7,8,9]:
    y = np.where(train[:,0] == digit, 1, -1)
    w = (Z.transpose().dot(Z)+penalty*np.identity(Z.shape[1])).I.dot(Z.transpose()).dot(y)
    w_transform = (Z_transform.transpose().dot(Z_transform)+penalty*np.identity(Z_transform.shape[1])).I.dot(Z_transform.transpose()).dot(y)
    y_out = np.where(test[:,0] == digit, 1, -1)    
    y_hat = np.sign(w.dot(Z_out.transpose()))
    Eout = 1-np.mean(y_hat == y_out)    
    y_hat_transform = np.sign(w_transform.dot(Z_out_transform.transpose()))
    Eout_transform = 1-np.mean(y_hat_transform == y_out)    
    Eouts[digit] = Eout  
    Eouts_transform[digit] = Eout_transform
    
# 1 vs. 5
X_15 = train[np.isin(train[:,0], [1,5]),1:3]
y_15 = np.where(train[np.isin(train[:,0], [1,5]),0] == 1, 1, -1)
N_15 = X_15.shape[0]
Z_15 = np.matrix(np.vstack(([1]*N_15,X_15[:,0],X_15[:,1],X_15[:,0]*X_15[:,1],X_15[:,0]**2,X_15[:,1]**2)).T)    
X_15_out = test[np.isin(test[:,0], [1,5]), 1:3]
y_15_out = np.where(test[np.isin(test[:,0], [1,5]),0] == 1, 1, -1)
N_15_out = X_15_out.shape[0]
Z_15_out = np.matrix(np.vstack(([1]*N_15_out,X_15_out[:,0],X_15_out[:,1],X_15_out[:,0]*X_15_out[:,1],X_15_out[:,0]**2,X_15_out[:,1]**2)).T)    

Eins = {}
Eouts = {}

for penalty in [0.01, 1]:
    w = (Z_15.transpose().dot(Z_15)+penalty*np.identity(Z_15.shape[1])).I.dot(Z_15.transpose()).dot(y_15)
    y_hat = np.sign(w.dot(Z_15.transpose()))
    Eins[penalty] = 1-np.mean(y_hat == y_15)
    y_hat_out =  np.sign(w.dot(Z_15_out.transpose()))
    Eouts[penalty] = 1-np.mean(y_hat_out == y_15_out)


# problem 10
X = np.array([
        [1,0],
        [0,1],
        [0,-1],
        [-1,0],
        [0,2],
        [0,-2],
        [-2,0]])
y = [-1,-1,-1,1,1,1,1]

reds = [X[i] for i in range(0, 7) if y[i] == -1]
greens= [X[i] for i in range(0, 7) if y[i] == 1]

# for visualization purposes
plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
plt.xlim(-3, 3)
plt.ylim(-3, 3)    
plt.show() 

def z(x):
    return np.array([x[1]**2-2*x[0]-1,x[0]**2-2*x[1]+1])

Z = np.apply_along_axis(z, 1, X)


reds = [Z[i] for i in range(0, 7) if y[i] == -1]
greens= [Z[i] for i in range(0, 7) if y[i] == 1]

# for visualization purposes
plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
plt.xlim(-6, 6)
plt.ylim(-6, 6)    
plt.show() 

#Z = np.matrix(np.apply_along_axis(lambda x: np.append(1,x), 1, Z))
Z = np.matrix(Z)

y = np.asarray(y, dtype=float).reshape(-1,1)

def K(x1,x2):
    return (1+x1.dot(x2))**2

H = np.zeros((7,7))
N = 7

for i in range(0,7):
    for j in range(0,7):
        H[i,j] = y[i]*y[j]*K(X[i],X[j])
      
import cvxopt
import quadprog 
from cvxopt import matrix as cvxopt_matrix  
from cvxopt import solvers as cvxopt_solvers     
        
#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((N, 1)))
G = cvxopt_matrix(-np.eye(N))
h = cvxopt_matrix(np.zeros(N))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))
 

cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])
alphas[alphas < 10e-10] = 0        
        
n_support_vectors = (alphas>0).sum()            

# RBF
N = 100

def target(x):
    return np.sign(x[1]-x[0]+0.25*np.sin(np.pi*x[0]))

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))

def svm_yhat(x, X, y, gamma, alphas, b):
    N = alphas.shape[0]
    return int(np.sign(np.sum([alphas[n][0]*y[n]*rbf_kernel(X[n],x,gamma) for n in range(0,N)])+b))

def regular_rbf_yhat(x, gamma, mus, w, b):
    return int(np.sign(np.sum([w[k]*np.exp(-gamma*np.linalg.norm(x-mus[k])**2) for k in range(0,K)])+b))

#problem 13 - 18

gamma = 1.5
K = 12

T = 100
total_runs = 0
c = 0
n_unseparable = 0

Eouts_svm = []
Eouts_regular_rbf = []

while c < T:
    
    print('starting run, total_runs=' +str(total_runs) + ', c=' +str(c))

    total_runs = total_runs + 1
    
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y = np.array([target(x_i) for x_i in X])
    X_out = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y_out = np.array([target(x_i) for x_i in X_out])
    
    #reds = [X[i] for i in range(0, N) if y[i] == -1]
    #greens= [X[i] for i in range(0, N) if y[i] == 1]
    #
    ## for visualization purposes
    #plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
    #plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)    
    #plt.show() 
    
    # svm
    
    H = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            H[i,j] = y[i]*y[j]*rbf_kernel(X[i],X[j],gamma)
            
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((N, 1)))
    G = cvxopt_matrix(-np.eye(N))
    h = cvxopt_matrix(np.zeros(N))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1)) 
    
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10    
    
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas < 10e-5] = 0
    
    w = np.zeros(X.shape[1])
    for n in range(0, N):
        w = w + alphas[n]*y[n]*X[n]           
    sv = np.argwhere(alphas>0)[0][0]
    b = y[sv] - np.sum([alphas[n][0]*y[n]*rbf_kernel(X[n],X[sv],gamma) for n in range(0,N)])
        
    yhat_svm = [svm_yhat(X[i], X, y, gamma, alphas, b) for i in range(0, N)]    
    misses_svm = [i for i in range(0, N) if yhat_svm[i] != y[i]]     
    Ein_svm = len(misses_svm)/N  

    if Ein_svm > 0:
        n_unseparable = n_unseparable+1
        print("unseparable data!")
    else:        
        
        yhat_out_svm = [svm_yhat(X_out[i], X_out, y_out, gamma, alphas, b) for i in range(0, N)]    
        misses_out_svm = [i for i in range(0, N) if yhat_out_svm[i] != y_out[i]]     
        Eout_svm = len(misses_out_svm)/N
        
        # regular rbf
    
        mus = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, K)])#X[np.random.choice(N, size=K, replace=False)]   
        S = np.zeros(N)
        
        cont = True
        
        while cont is True:
        
            for i in range(0,N):
                dists = [np.linalg.norm(X[i]-mu) for mu in mus]
                S[i] = np.nanargmin(dists)
                
            prev_mus = mus.copy()
                
            for k in range(0, K):
                inds = np.where(S==k)
                if len(np.where(S==k)[0]) > 0:
                    mus[k] = np.array([np.mean(X[inds,0]),np.mean(X[inds,1])])
                else:
                    mus[k] = np.array([None,None])
                    cont = False
            
            if np.array_equal(mus, prev_mus):
                cont = False
        
        cluster_sizes = []
        
        for k in range(0, K):
            cluster_sizes.append(len(np.where(S==k)[0]))
                
        if 0 in cluster_sizes:
            print("empty cluster!")
        else:
        
    #    for k in range(0, K):
    #        pts = X[np.where(S==k)]
    #        plt.scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], marker='.')
    #    
    #    plt.scatter(x=[mu[0] for mu in mus], y=[mu[1] for mu in mus], marker='o', color='black')
    #    plt.xlim(-1, 1)
    #    plt.ylim(-1, 1)    
    #    plt.show() 
        
            phi = np.matrix(np.zeros((N,K+1)))
            phi[:,0] = 1
            for i in range(0,N):
                for k in range(0,K):
                    phi[i,k+1] = rbf_kernel(X[i],mus[k],gamma)
            
            w = np.array(phi.transpose().dot(phi).I.dot(phi.transpose()).dot(y).transpose())
            b = w[0]
            w = w[1:]
                    
            yhat_out_regular_rbf = [regular_rbf_yhat(X_out[i], gamma, mus, w, b) for i in range(0, N)]            
            misses_out_regular_rbf = [i for i in range(0, N) if yhat_out_regular_rbf[i] != y_out[i]]     
            Eout_regular_rbf = len(misses_out_regular_rbf)/N 
            
            Eouts_svm.append(Eout_svm)
            Eouts_regular_rbf.append(Eout_regular_rbf)
            
            c = c+1

print("pct unseperable: " + str(n_unseparable/total_runs))

n_kernel_wins = len([i for i in range(0,T) if Eouts_svm[i] < Eouts_regular_rbf[i]])
print('pct kernel wins with K=9: ' + str(n_kernel_wins/T)) # 0.02 with K = 9
# in problem 13 - we had 0 unseparable cases!


# K=9 vs. K=12
gamma = 1.5

T = 100
total_runs = 0
c = 0

Eins_k9 = []
Eouts_k9 = []
Eins_k12 = []
Eouts_k12 = []

while c < T:
    
    print('starting run, total_runs=' +str(total_runs) + ', c=' +str(c))

    total_runs = total_runs + 1
    
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y = np.array([target(x_i) for x_i in X])
    X_out = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y_out = np.array([target(x_i) for x_i in X_out])

    K = 9

    mus = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, K)])#X[np.random.choice(N, size=K, replace=False)]   
    S = np.zeros(N)
    
    cont = True
    
    while cont is True:
    
        for i in range(0,N):
            dists = [np.linalg.norm(X[i]-mu) for mu in mus]
            S[i] = np.nanargmin(dists)
            
        prev_mus = mus.copy()
            
        for k in range(0, K):
            inds = np.where(S==k)
            if len(np.where(S==k)[0]) > 0:
                mus[k] = np.array([np.mean(X[inds,0]),np.mean(X[inds,1])])
            else:
                mus[k] = np.array([None,None])
                cont = False
        
        if np.array_equal(mus, prev_mus):
            cont = False
    
    cluster_sizes = []
    
    for k in range(0, K):
        cluster_sizes.append(len(np.where(S==k)[0]))
            
    if 0 in cluster_sizes:
        print("empty cluster!")
    else:
    
        phi = np.matrix(np.zeros((N,K+1)))
        phi[:,0] = 1
        for i in range(0,N):
            for k in range(0,K):
                phi[i,k+1] = rbf_kernel(X[i],mus[k],gamma)
        
        w = np.array(phi.transpose().dot(phi).I.dot(phi.transpose()).dot(y).transpose())
        b = w[0]
        w = w[1:]
        
        yhat = [regular_rbf_yhat(X[i], gamma, mus, w, b) for i in range(0, N)]
        misses = [i for i in range(0, N) if yhat[i] != y[i]]   
        Ein_k9 = len(misses)/N
#        Eins_k9.append(Ein)
        yhat_out = [regular_rbf_yhat(X_out[i], gamma, mus, w, b) for i in range(0, N)]            
        misses_out = [i for i in range(0, N) if yhat_out[i] != y_out[i]]     
        Eout_k9 = len(misses_out)/N
#        Eouts_k9.append(Eout)
        
        K = 12
        
        mus = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, K)])#X[np.random.choice(N, size=K, replace=False)]   
        S = np.zeros(N)
        
        cont = True
        
        while cont is True:
        
            for i in range(0,N):
                dists = [np.linalg.norm(X[i]-mu) for mu in mus]
                S[i] = np.nanargmin(dists)
                
            prev_mus = mus.copy()
                
            for k in range(0, K):
                inds = np.where(S==k)
                if len(np.where(S==k)[0]) > 0:
                    mus[k] = np.array([np.mean(X[inds,0]),np.mean(X[inds,1])])
                else:
                    mus[k] = np.array([None,None])
                    cont = False
            
            if np.array_equal(mus, prev_mus):
                cont = False
        
        cluster_sizes = []
        
        for k in range(0, K):
            cluster_sizes.append(len(np.where(S==k)[0]))
                
        if 0 in cluster_sizes:
            print("empty cluster!")
        else:
        
            phi = np.matrix(np.zeros((N,K+1)))
            phi[:,0] = 1
            for i in range(0,N):
                for k in range(0,K):
                    phi[i,k+1] = rbf_kernel(X[i],mus[k],gamma)
            
            w = np.array(phi.transpose().dot(phi).I.dot(phi.transpose()).dot(y).transpose())
            b = w[0]
            w = w[1:]
            
            yhat = [regular_rbf_yhat(X[i], gamma, mus, w, b) for i in range(0, N)]
            misses = [i for i in range(0, N) if yhat[i] != y[i]]   
            Ein_k12 = len(misses)/N
            yhat_out = [regular_rbf_yhat(X_out[i], gamma, mus, w, b) for i in range(0, N)]            
            misses_out = [i for i in range(0, N) if yhat_out[i] != y_out[i]]     
            Eout_k12 = len(misses_out)/N
                
            Eins_k9.append(Ein_k9)        
            Eouts_k9.append(Eout_k9)
            Eins_k12.append(Ein_k12)        
            Eouts_k12.append(Eout_k12)            
            
            c = c+1

n_ans_a = len([i for i in range(0,N) if (Eins_k12[i] < Eins_k9[i]) & (Eouts_k12[i] > Eouts_k9[i])])
n_ans_b = len([i for i in range(0,N) if (Eins_k12[i] > Eins_k9[i]) & (Eouts_k12[i] < Eouts_k9[i])])        
n_ans_c = len([i for i in range(0,N) if (Eins_k12[i] > Eins_k9[i]) & (Eouts_k12[i] > Eouts_k9[i])])        
n_ans_d = len([i for i in range(0,N) if (Eins_k12[i] < Eins_k9[i]) & (Eouts_k12[i] < Eouts_k9[i])])        
n_ans_e = len([i for i in range(0,N) if (Eins_k12[i] == Eins_k9[i]) & (Eouts_k12[i] == Eouts_k9[i])])        

# gamma = 1.5 vs gamma = 2

K = 9

T = 100
total_runs = 0
c = 0

Eins_gamma1pt5 = []
Eouts_gamma1pt5 = []
Eins_gamma2 = []
Eouts_gamma2 = []

while c < T:
    
    print('starting run, total_runs=' +str(total_runs) + ', c=' +str(c))

    total_runs = total_runs + 1
    
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y = np.array([target(x_i) for x_i in X])
    X_out = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
    y_out = np.array([target(x_i) for x_i in X_out])

    mus = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, K)])#X[np.random.choice(N, size=K, replace=False)]   
    S = np.zeros(N)
    
    cont = True
    
    while cont is True:
    
        for i in range(0,N):
            dists = [np.linalg.norm(X[i]-mu) for mu in mus]
            S[i] = np.nanargmin(dists)
            
        prev_mus = mus.copy()
            
        for k in range(0, K):
            inds = np.where(S==k)
            if len(np.where(S==k)[0]) > 0:
                mus[k] = np.array([np.mean(X[inds,0]),np.mean(X[inds,1])])
            else:
                mus[k] = np.array([None,None])
                cont = False
        
        if np.array_equal(mus, prev_mus):
            cont = False
    
    cluster_sizes = []
    
    for k in range(0, K):
        cluster_sizes.append(len(np.where(S==k)[0]))
            
    if 0 in cluster_sizes:
        print("empty cluster!")
    else:
        
        gamma = 1.5
    
        phi = np.matrix(np.zeros((N,K+1)))
        phi[:,0] = 1
        for i in range(0,N):
            for k in range(0,K):
                phi[i,k+1] = rbf_kernel(X[i],mus[k],gamma)
        
        w = np.array(phi.transpose().dot(phi).I.dot(phi.transpose()).dot(y).transpose())
        b = w[0]
        w = w[1:]
        
        yhat = [regular_rbf_yhat(X[i], gamma, mus, w, b) for i in range(0, N)]
        misses = [i for i in range(0, N) if yhat[i] != y[i]]   
        Ein_gamma1pt5 = len(misses)/N
        yhat_out = [regular_rbf_yhat(X_out[i], gamma, mus, w, b) for i in range(0, N)]            
        misses_out = [i for i in range(0, N) if yhat_out[i] != y_out[i]]     
        Eout_gamma1pt5 = len(misses_out)/N
#        Eouts_k9.append(Eout)
        
        gamma = 2                
        
        phi = np.matrix(np.zeros((N,K+1)))
        phi[:,0] = 1
        for i in range(0,N):
            for k in range(0,K):
                phi[i,k+1] = rbf_kernel(X[i],mus[k],gamma)
        
        w = np.array(phi.transpose().dot(phi).I.dot(phi.transpose()).dot(y).transpose())
        b = w[0]
        w = w[1:]
        
        yhat = [regular_rbf_yhat(X[i], gamma, mus, w, b) for i in range(0, N)]
        misses = [i for i in range(0, N) if yhat[i] != y[i]]   
        Ein_gamma2 = len(misses)/N
        yhat_out = [regular_rbf_yhat(X_out[i], gamma, mus, w, b) for i in range(0, N)]            
        misses_out = [i for i in range(0, N) if yhat_out[i] != y_out[i]]     
        Eout_gamma2 = len(misses_out)/N
            
        Eins_gamma1pt5.append(Ein_gamma1pt5)        
        Eouts_gamma1pt5.append(Eout_gamma1pt5)
        Eins_gamma2.append(Ein_gamma2)        
        Eouts_gamma2.append(Eout_gamma2)            
            
        c = c+1
        
n_ans_a = len([i for i in range(0,N) if (Eins_gamma2[i] < Eins_gamma1pt5[i]) & (Eouts_gamma2[i] > Eouts_gamma1pt5[i])])
n_ans_b = len([i for i in range(0,N) if (Eins_gamma2[i] > Eins_gamma1pt5[i]) & (Eouts_gamma2[i] < Eouts_gamma1pt5[i])])        
n_ans_c = len([i for i in range(0,N) if (Eins_gamma2[i] > Eins_gamma1pt5[i]) & (Eouts_gamma2[i] > Eouts_gamma1pt5[i])])        
n_ans_d = len([i for i in range(0,N) if (Eins_gamma2[i] < Eins_gamma1pt5[i]) & (Eouts_gamma2[i] < Eouts_gamma1pt5[i])])        
n_ans_e = len([i for i in range(0,N) if (Eins_gamma2[i] == Eins_gamma1pt5[i]) & (Eouts_gamma2[i] == Eouts_gamma1pt5[i])])        
       
# pct of time Ein = 0 for regular rbf with K =9 and gamma=1.5
pct_rbf_ein_zero = len([i for i in range(0,N) if Eins_k9[i] == 0])/T
