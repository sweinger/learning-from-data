#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:14:28 2020

@author: sweinger
"""

a_list = []

for t in range(0, 10000):
    
    x = [random.uniform(-1, 1), random.uniform(-1, 1)]
    y = [np.sin(np.pi*x[0]), np.sin(np.pi*x[1])]
    a = (y[0]*x[0]+y[1]*x[1])/(x[0]**2+x[1]**2)
    a_list.append(a)

a_hat = np.mean(a_list)     

X = [random.uniform(-1, 1) for t in range(0, 1000)]
bias = np.mean([(a_hat*x-np.sin(np.pi*x))**2 for x in X])
print(bias)

all_deviations = []

for t in X:
    all_deviations.append(np.mean([(t*a-t*a_hat)**2 for a in a_list]))

variance = np.mean(all_deviations)
print(variance)

oos_error = bias + variance


# 7.a constant
a_list = []

for t in range(0, 10000):
    
    x = [random.uniform(-1, 1), random.uniform(-1, 1)]
    y = [np.sin(np.pi*x[0]), np.sin(np.pi*x[1])]
    a = (y[0]+y[1])/2
    a_list.append(a)

a_hat = np.mean(a_list)  

X = [random.uniform(-1, 1) for t in range(0, 1000)]
bias = np.mean([(a_hat*x-np.sin(np.pi*x))**2 for x in X])

all_deviations = []

for t in X:
    all_deviations.append(np.mean([(t*a-t*a_hat)**2 for a in a_list]))

variance = np.mean(all_deviations)

oos_error = bias + variance
print(oos_error) # 0.591735071570711

#7.b (above) 0.5191106117322595

#7.c linear model
w_list = []

for t in range(0, 10000):
    
    x = np.matrix([(1, random.uniform(-1, 1)), (1, random.uniform(-1, 1))])
    y = np.sin(np.pi*x[:,1])
    w = (x.transpose().dot(x)).I.dot(x.transpose()).dot(y)
    w_list.append(w)
    
w_bar = np.matrix([np.mean([w[0] for w in w_list]), np.mean([w[1] for w in w_list])]).transpose()
    
X = np.matrix([(1, random.uniform(-1, 1)) for i in range(0, 1000)])

bias = np.mean((np.asarray(np.sin(np.pi*X[:,1]))-np.asarray(X.dot(w_bar)))**2)

all_deviations = []

for t in X:
    all_deviations.append(np.mean([(t.dot(w)-t.dot(w_bar))**2 for w in w_list]))
    
variance = np.mean(all_deviations)  

oos_error = bias + variance
print(oos_error)  # 1.9010671163360855

#7.d quadratic transformation zero intercept
a_list = []

for t in range(0, 10000):
    
    x = [random.uniform(-1, 1), random.uniform(-1, 1)]
    y = [np.sin(np.pi*x[0]), np.sin(np.pi*x[1])]
    a = (y[0]*x[0]**2+y[1]*x[1]**2)/(x[0]**4+x[1]**4)
    a_list.append(a)

a_hat = np.mean(a_list)     

X = [random.uniform(-1, 1) for t in range(0, 1000)]
bias = np.mean([(a_hat*x**2-np.sin(np.pi*x))**2 for x in X])
print(bias)

all_deviations = []

for t in X:
    z = t**2
    all_deviations.append(np.mean([(z*a-z*a_hat)**2 for a in a_list]))

variance = np.mean(all_deviations)
print(variance)

oos_error = bias + variance
print(oos_error) # 12.135664676211784

#7.e linear model with quadratic transformation
w_list = []

for t in range(0, 10000):
    
    x = np.matrix([(1, random.uniform(-1, 1)), (1, random.uniform(-1, 1))])
    y = np.sin(np.pi*x[:,1])
    z = np.square(x)
    w = (z.transpose().dot(z)).I.dot(z.transpose()).dot(y)
    w_list.append(w)
    
w_bar = np.matrix([np.mean([w[0] for w in w_list]), np.mean([w[1] for w in w_list])]).transpose()
    
X = np.matrix([(1, random.uniform(-1, 1)) for i in range(0, 1000)])

bias = np.mean((np.asarray(np.sin(np.pi*X[:,1]))-np.asarray(np.square(X).dot(w_bar)))**2)

all_deviations = []

for t in X:
    z = np.square(t)
    all_deviations.append(np.mean([(z.dot(w)-z.dot(w_bar))**2 for w in w_list]))
    
variance = np.mean(all_deviations)  

oos_error = bias + variance
print(oos_error) # 662071.9506727979
