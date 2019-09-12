#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complementary functions to:
    - create model matrices and heredity relation matrix
    - calculate combinations of n in r
    - save objects in pickle
    - fit a model
"""
import itertools 
import math # used by 'nchoosek' function
import numpy as np
import pickle
from sklearn import linear_model

# Classes ---------------------------------------------------------------------
class ModMat(object):
    """
    Python class for model matrices.
    """
    def __init__(self, x, labels, relmat):
        self.x = x
        self.labels = labels
        self.relmat = relmat

# Functions -------------------------------------------------------------------
def save_object(obj, filename):
    """ Save object as pickle object in 'filename'.
    Input:
        obj: object (ModMat class, numpy array, etc.).
        filename: directory and name of the file (str).    
    """
    
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def nchoosek(n,r): 
    """ Compute the total number of combinations of 'n' in 'r'.
    """
    
    f = math.factorial
    res = f(n) / f(r) / f(n-r)
    return int(res)

def twofimodmat( D ):
    """ Construct two-factor interaction model matrix . 
    Input:
        D: matrix (numpy array, float).
    Output:
        Model matrix including main effect and two-factor 
        interaction columns, effect labels and heredity relationship matix 
        (ModMat class). 
    """
    
    (N,m) = np.shape(D)
    ncombtwo = itertools.combinations(range(m), 2) 
    nm = nchoosek(m, 2)
    Matt = np.zeros([N, nm])
    c = 0
    
    # Create relationship matrix
    fullsize = m+1
    fullsize = fullsize + nchoosek(fullsize-1,2)
    R = np.zeros((m+1,fullsize), dtype = int)
    R[0, 0:(m+1)] = np.ones((1, m+1), dtype = int)
    lec = m+2
    Combos = itertools.combinations(range(1, lec-1), 2)
    cc = lec - 1
    for i, j in Combos:
        R[i, cc] = 1
        R[j, cc] = 1
        cc = cc + 1 
    # Remove row and column for the intercept
    R = np.delete(R, 0, axis = 0)
    R = np.delete(R, 0, axis = 1)
    
    # Create labels for effects
    alphabet = []
    for letter in range(65,91):
        alphabet.append(chr(letter))
    # Labels for the MEs
    factors = alphabet[0:m]
    labels = alphabet[0:m] 
    
    # Construct two-factor interaction matrix
    for i,j in ncombtwo:
        Matt[:, c] = D[:, i]*D[:, j]
        # Labels for 2FIs
        lab_int = ':'.join([factors[i], factors[j]])
        labels.append(lab_int)  
        c = c + 1
    
    X = np.concatenate((D,Matt), axis = 1)
    return ModMat(X, labels, R)

def quadmodmat( D ):
    """ Construct full quadratic model matrix.    
    Input:
        D: design matrix (numpy array, float).
    Output:
        Model matrix including linear effect, two-factor interaction and 
        quadratic effect columns, effect labels and heredity relationship matix 
        (ModMat class).
    """    
    
    (N,m) = np.shape(D)
    ncombtwo = itertools.combinations(range(m), 2) 
    nm = nchoosek(m, 2)
    Matt = np.zeros([N, nm])
    c = 0
    
    # Create relationship matrix
    fullsize = m+1
    fullsize = fullsize + nchoosek(fullsize-1,2) + m
    R = np.zeros((m+1,fullsize), dtype = int)
    R[0, 0:(m+1)] = np.ones((1, m+1), dtype = int)
    lec = m+2
    Combos = itertools.combinations(range(1, lec-1), 2)
    cc = lec - 1
    for i, j in Combos:
        R[i, cc] = 1
        R[j, cc] = 1
        cc = cc + 1 
        
    R[1:,range((m+1+nm), fullsize)] = 2*np.eye(m) 
    
    # Remove row and column for the intercept
    R = np.delete(R, 0, axis = 0)
    R = np.delete(R, 0, axis = 1)
    
    # Create labels for effects
    alphabet = []
    for letter in range(65,91):
        alphabet.append(chr(letter))
    # Labels for the LEs
    factors = alphabet[0:m]
    labels = alphabet[0:m] 
    
    # Construct quadratic model matrix
    for i, j in ncombtwo:
        Matt[:, c] = D[:, i]*D[:, j]
        # Labels for 2FIs
        lab_int = ':'.join([factors[i], factors[j]])
        labels.append(lab_int)         
        c = c + 1
    
    # Labels for quadratic effects 
    for i in range(m):
        lab_quad = factors[i]+'.2'
        labels.append(lab_quad)  
    
    Dsq = D**2
    X = np.concatenate((D,Matt,Dsq), axis = 1)
    return ModMat(X, labels, R)

def fit_model_rss(X, Y, selected):
    """ Fit a model and extract its residual sum of squares.
    Inputs:
        X: N x p model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Y: N x 1 vector of responses (numpy array, float).             
        selected: 1 x p vector of selected effects (numpy array, binary).
    Outputs:
        rss: Residual sum of squared of fitted model. 
    """
    
    number_of_selected = np.sum(selected)
    if number_of_selected > 0:
	    Xsub = X[:,selected ==1]
		# Create linear regression object
	    regr = linear_model.LinearRegression()
		# Fit model
	    regr.fit(Xsub, Y)
		# Obtain predicted values
	    Ypred = regr.predict(Xsub)
		# Compute the residual sum of squares
	    rss = ((Y - Ypred) ** 2).sum() 
    else :
	    rss = (Y ** 2).sum() 
    
    return rss
