#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:37:35 2020

@author: sweinger
"""

import cvxopt
import quadprog

train = np.loadtxt('/Users/sweinger/Documents/caltech_course/in.dta.txt')
test = np.loadtxt('/Users/sweinger/Documents/caltech_course/out.dta.txt')

X_train = train[0:25,0:2]
Y_train = train[0:25,2]
X_validation = train[25:,0:2]
Y_validation = train[25:,2]
X_out = test[:,0:2]
Y_out = test[:,2]

N = X_train.shape[0]
N_validation= X_validation.shape[0]
N_out = X_out.shape[0]

reds = [X_train[i] for i in range(0, N) if Y_train[i] == -1]
greens= [X_train[i] for i in range(0, N) if Y_train[i] == 1]

# for visualization purposes
plt.scatter(x=[r[0] for r in reds], y=[r[1] for r in reds], color='red')
plt.scatter(x=[g[0] for g in greens], y=[g[1] for g in greens], color='green')
plt.xlim(-1, 1)
plt.ylim(-1, 1)    
plt.show()  

Z_train = np.matrix(np.vstack(([1]*N,X_train[:,0],X_train[:,1],X_train[:,0]**2,X_train[:,1]**2,X_train[:,0]*X_train[:,1],np.abs((X_train[:,0]-X_train[:,1])),np.abs((X_train[:,0]+X_train[:,1])))).T)
Z_validation = np.matrix(np.vstack(([1]*N_validation,X_validation[:,0],X_validation[:,1],X_validation[:,0]**2,X_validation[:,1]**2,X_validation[:,0]*X_validation[:,1],np.abs((X_validation[:,0]-X_validation[:,1])),np.abs((X_validation[:,0]+X_validation[:,1])))).T)
Z_out = np.matrix(np.vstack(([1]*N_out,X_out[:,0],X_out[:,1],X_out[:,0]**2,X_out[:,1]**2,X_out[:,0]*X_out[:,1],np.abs((X_out[:,0]-X_out[:,1])),np.abs((X_out[:,0]+X_out[:,1])))).T)

def classification_errors(Z_train, Y_train, Z_test, Y_test, k):
    Z = Z_train[:,0:k+1]
    w = (Z.transpose().dot(Z)).I.dot(Z.transpose()).dot(Y_train)
    y_hat = np.sign(w.dot(Z_test[:,0:k+1].transpose()))
    E = 1-np.mean(y_hat == Y_test)    
    return(E)    


for k in range(3,8):
    print('k:' + str(k) + ', error: ' +str(classification_errors(Z_train, Y_train, Z_validation, Y_validation, k)))


for k in range(3,8):
    print('k:' + str(k) + ', error: ' +str(classification_errors(Z_train, Y_train, Z_out, Y_out, k)))


for k in range(3,8):
    print('k:' + str(k) + ', error: ' +str(classification_errors(Z_validation, Y_validation, Z_train, Y_train, k)))
    
    
for k in range(3,8):
    print('k:' + str(k) + ', error: ' +str(classification_errors(Z_validation, Y_validation, Z_out, Y_out, k)))    
    

# PLA vs. SVM
    
N = 10

d_pla = []
d_svm = []
n_svs = []

for B in range(0, 1000):
   
    while True:     
        f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
        
        x = np.linspace(-1, 1, 100)
        m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
        b = f[0][1] - m*f[0][0]
        y = m*x+b    
        
        X = np.matrix([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
        Y = np.matrix([-1 if x_i[:,1] < (m*x_i[:,0]+b) else 1 for x_i in X])
        
        s = (Y == 1).sum()
        if (s != 0) & (s != N):
            break
        else:
            cont = False    
    
    #check if all inouts are on same side of the line!
    
    #reds = [X[i] for i in range(0, N) if Y[:,i][0,0] == -1]
    #greens= [X[i] for i in range(0, N) if Y[:,i][0,0] == 1]
    #
    #plt.scatter(x=[r[:,0] for r in reds], y=[r[:,1] for r in reds], color='red')
    #plt.scatter(x=[g[:,0] for g in greens], y=[g[:,1] for g in greens], color='green')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)    
    #plt.show() 
    
    # PLA
    W = [0, 0, 0] 
    
    for t in range(0, 10000):
        H = [int(np.sign(W[0]+W[1]*X[i][:,0]+W[2]*X[i][:,1])) for i in range(0, N)]
        misses = [i for i in range(0, N) if H[i] != Y[:,i][0,0]]
        if len(misses) == 0:
            break;
        p = random.choice(misses)
        W[0] = W[0] + Y[:,p][0,0]*1
        W[1] = (W[1] + Y[:,p][0,0]*X[p][:,0])[0,0]
        W[2] = (W[2] + Y[:,p][0,0]*X[p][:,1])[0,0]
        
    N_out = 1000
    X_out = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N_out)]
    Y_out = [-1 if x_i[1] < (m*x_i[0]+b) else 1 for x_i in X_out]    
    
    H_out = [int(np.sign(W[0]+W[1]*X_out[i][0]+W[2]*X_out[i][1])) for i in range(0, N_out)]
    misses = [i for i in range(0, N_out) if H_out[i] != Y_out[i]]    
    disagreement = len(misses)/N_out
    d_pla.append(disagreement)
    
    # SVM
    
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=float).reshape(-1,1)
    X_dash = Y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((N, 1)))
    G = cvxopt_matrix(-np.eye(N))
    h = cvxopt_matrix(np.zeros(N))
    A = cvxopt_matrix(Y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
     
    
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas < 10e-10] = 0
        
    w = np.zeros(X.shape[1])
    for n in range(0, N):
        w = w + alphas[n]*Y[n]*X[n]
        
    n_support_vectors = (alphas>0).sum()    
    sv = np.argwhere(alphas>0)[0][0]
    b = 1/Y[sv][0] - w.dot(X[sv])
    
    H_svm_out = [int(np.sign(w.dot(X_out[i])+b)) for i in range(0, N_out)]
    misses_svm = [i for i in range(0, N_out) if H_svm_out[i] != Y_out[i]]    
    disagreement_svm = len(misses_svm)/N_out
    d_svm.append(disagreement_svm)
    n_svs.append(n_support_vectors)

print("SVM is better than PLA: ")
print((np.array(d_svm) < np.array(d_pla)).mean())


N = 100

d_pla = []
d_svm = []
n_svs = []

for B in range(0, 1000):
    
    if B % 100 == 0:
        print(B)
   
    while True:     
        f = ((random.uniform(-1, 1), random.uniform(-1, 1)), (random.uniform(-1, 1), random.uniform(-1, 1)))
        
        x = np.linspace(-1, 1, 100)
        m = (f[1][1]-f[0][1])/(f[1][0]-f[0][0])
        b = f[0][1] - m*f[0][0]
        y = m*x+b    
        
        X = np.matrix([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N)])
        Y = np.matrix([-1 if x_i[:,1] < (m*x_i[:,0]+b) else 1 for x_i in X])
        
        s = (Y == 1).sum()
        if (s != 0) & (s != N):
            break
        else:
            cont = False    
    
    #check if all inouts are on same side of the line!
    
    #reds = [X[i] for i in range(0, N) if Y[:,i][0,0] == -1]
    #greens= [X[i] for i in range(0, N) if Y[:,i][0,0] == 1]
    #
    #plt.scatter(x=[r[:,0] for r in reds], y=[r[:,1] for r in reds], color='red')
    #plt.scatter(x=[g[:,0] for g in greens], y=[g[:,1] for g in greens], color='green')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)    
    #plt.show() 
    
    # PLA
    W = [0, 0, 0] 
    
    for t in range(0, 10000):
        H = [int(np.sign(W[0]+W[1]*X[i][:,0]+W[2]*X[i][:,1])) for i in range(0, N)]
        misses = [i for i in range(0, N) if H[i] != Y[:,i][0,0]]
        if len(misses) == 0:
            break;
        p = random.choice(misses)
        W[0] = W[0] + Y[:,p][0,0]*1
        W[1] = (W[1] + Y[:,p][0,0]*X[p][:,0])[0,0]
        W[2] = (W[2] + Y[:,p][0,0]*X[p][:,1])[0,0]
        
    N_out = 1000
    X_out = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(0, N_out)]
    Y_out = [-1 if x_i[1] < (m*x_i[0]+b) else 1 for x_i in X_out]    
    
    H_out = [int(np.sign(W[0]+W[1]*X_out[i][0]+W[2]*X_out[i][1])) for i in range(0, N_out)]
    misses = [i for i in range(0, N_out) if H_out[i] != Y_out[i]]    
    disagreement = len(misses)/N_out
    d_pla.append(disagreement)
    
    # SVM
    
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=float).reshape(-1,1)
    X_dash = Y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    
    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((N, 1)))
    G = cvxopt_matrix(-np.eye(N))
    h = cvxopt_matrix(np.zeros(N))
    A = cvxopt_matrix(Y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
     
    
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas < 10e-10] = 0
        
    w = np.zeros(X.shape[1])
    for n in range(0, N):
        w = w + alphas[n]*Y[n]*X[n]
        
    n_support_vectors = (alphas>0).sum()    
    sv = np.argwhere(alphas>0)[0][0]
    b = 1/Y[sv][0] - w.dot(X[sv])
    
    H_svm_out = [int(np.sign(w.dot(X_out[i])+b)) for i in range(0, N_out)]
    misses_svm = [i for i in range(0, N_out) if H_svm_out[i] != Y_out[i]]    
    disagreement_svm = len(misses_svm)/N_out
    d_svm.append(disagreement_svm)
    n_svs.append(n_support_vectors)

print("SVM is better than PLA: ")
print((np.array(d_svm) < np.array(d_pla)).mean())

print("Average number of support vectors:")
print(np.mean(n_svs))
