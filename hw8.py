#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:31:17 2020

@author: sweinger
"""
from sklearn import svm

train = np.loadtxt('/Users/sweinger/Documents/caltech_course/features.train.txt')
test = np.loadtxt('/Users/sweinger/Documents/caltech_course/features.test.txt')

# problem 2

X = train[:,1:3]
Eins = {}

for digit in [0,2,4,6,8]:
    y = np.where(train[:,0] == digit, 1, -1)
    clf = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1.0, coef0=1)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    Ein = (preds != y).mean()
    Eins[digit] = Ein

display(Eins)
print(max(Eins, key=Eins.get))

# max Ein is for 0
y = np.where(train[:,0] == 0, 1, -1)
clf_max = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_max.fit(X, y)

Eins = {}

for digit in [1,3,5,7,9]:
    y = np.where(train[:,0] == digit, 1, -1)
    clf = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1.0, coef0=1)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    Ein = (preds != y).mean()
    Eins[digit] = Ein

display(Eins)
print(min(Eins, key=Eins.get))

# min Ein is for 1
y = np.where(train[:,0] == 1, 1, -1)
clf_min = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_min.fit(X, y)

print(abs(sum(clf_max.n_support_)-sum(clf_min.n_support_)))

# 1 vs. 5

X_15 = train[np.isin(train[:,0], [1,5]),1:3]
y_15 = np.where(train[np.isin(train[:,0], [1,5]),0] == 1, 1, -1)
X_out = test[np.isin(test[:,0], [1,5]), 1:3]
y_out = test[np.isin(test[:,0], [1,5]), 0]
n_svs = []
Eins = []
Eouts = []

for C in [0.001, 0.01, 0.1, 1]:
    clf_15 = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
    clf_15.fit(X_15, y_15)
    n_svs.append(sum(clf_15.n_support_))

    preds = clf_15.predict(X_15)
    preds_out = clf_15.predict(X_out)
    Eins.append((preds != y_15).mean())
    Eouts.append((preds_out != y_out).mean())
    
# Problem 6 
# C= 0.0001, Q =2 or 5, want Ein
C = 0.0001
clf_2 = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_2.fit(X_15,y_15)
clf_5 = svm.SVC(C=C, kernel='poly', degree=5, gamma=1.0, coef0=1)
clf_5.fit(X_15,y_15)

print((clf_2.predict(X_15) != y_15).mean())
print((clf_5.predict(X_15) != y_15).mean())

# C= 0.0001, Q =2 or 5, want num SVs
C = 0.001
clf_2 = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_2.fit(X_15,y_15)
clf_5 = svm.SVC(C=C, kernel='poly', degree=5, gamma=1.0, coef0=1)
clf_5.fit(X_15,y_15)

print(sum(clf_2.n_support_))
print(sum(clf_5.n_support_))

# C= 0.01, Q =2 or 5, want Ein
C = 0.01
clf_2 = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_2.fit(X_15,y_15)
clf_5 = svm.SVC(C=C, kernel='poly', degree=5, gamma=1.0, coef0=1)
clf_5.fit(X_15,y_15)

print((clf_2.predict(X_15) != y_15).mean())
print((clf_5.predict(X_15) != y_15).mean())

# C= 1, Q =2 or 5, want Eout
C = 1.0
clf_2 = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
clf_2.fit(X_15,y_15)
clf_5 = svm.SVC(C=C, kernel='poly', degree=5, gamma=1.0, coef0=1)
clf_5.fit(X_15,y_15)

print((clf_2.predict(X_out) != y_out).mean())
print((clf_5.predict(X_out) != y_out).mean())

# Cross validation

Cs = [0.0001, 0.001, 0.01, 0.1, 1]
choices = {C:0 for C in Cs}
Ecvs_choices = {C:[] for C in Cs}
for T in range(0, 100):

    train_cv = train.copy()
    train_cv = train_cv[np.isin(train_cv[:,0], [1,5]),:]
    np.random.shuffle(train_cv)
    cv_folds = np.array_split(train_cv, 10)
    Ecvs = {}
    
    for C in Cs:    
        Eouts = []
        for k in range(0, 10):
            train_fold = np.concatenate([cv_folds[x] for x in range(0, 10) if x != k])
            X_fold = train_fold[:,1:3]
            y_fold = np.where(train_fold[:,0] == 1, 1, -1)
            test_fold = cv_folds[k]
            X_out = test_fold[:,1:3]
            y_out = np.where(test_fold[:,0] == 1, 1, -1)
            clf_cv = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0, coef0=1)
            clf_cv.fit(X_fold, y_fold)
            preds_cv = clf_cv.predict(X_out)
            Eout = (preds_cv != y_out).mean()
            Eouts.append(Eout)
        Ecvs[C] = np.mean(Eouts)
    
    choice = min(Ecvs, key=Ecvs.get)
    choices[choice] = choices[choice] + 1
    for e in Ecvs:
        Ecvs_choices[e].append(Ecvs[e])

winner = max(choices, key=choices.get)
np.mean(Ecvs_choices[winner])

X_15 = train[np.isin(train[:,0], [1,5]),1:3]
y_15 = np.where(train[np.isin(train[:,0], [1,5]),0] == 1, 1, -1)
X_out = test[np.isin(test[:,0], [1,5]), 1:3]
y_out = test[np.isin(test[:,0], [1,5]), 0]

Cs = [0.01, 1, 100, 10**4, 10**6]
Eins = {}
Eouts = {}

for C in Cs:
    clf_rbf = svm.SVC(C=C, kernel='rbf', gamma=1.0)
    clf_rbf.fit(X_15, y_15)
    preds = clf_rbf.predict(X_15)
    preds_out = clf_rbf.predict(X_out)
    Eins[C] = (preds != y_15).mean()
    Eouts[C] = (preds_out != y_out).mean()
    
print(min(Eins, key=Eins.get))
print(min(Eouts, key=Eouts.get))