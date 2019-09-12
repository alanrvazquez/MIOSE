#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions containing first-order algorithms to compute good starting solutions
to the MIO problem and to specify the values of the constants in the boosting
constraints.    
"""
import numpy as np

class BetaStart(object):
    """
    Python class for solutions.
    """
    
    def __init__(self, b, val):
        self.b = b
        self.objval = val
        
class AdvancedStart(object):
    """
    Python class for advanced starting solutions.
    """      
    
    def __init__(self, bp, bm, z, maxb):
        self.bp = bp
        self.bm = bm
        self.z = z
        self.maxb = maxb        

# Functions -------------------------------------------------------------------
def gbeta(X, y, betap, N, p):
    """ Compute residual sum of squares (RSS).
    Input:
        X: N x p model marix, where N is the number of observations and p the number
           of parameters (numpy array, float).
        y: N x 1 vector of responses, where N is the number of observations 
           (numpy array, float).
        betap: p x 1 vector of estimated effects (numpy array, float).
        N: number of observations (int).
        p: number of parameters or effects (int).
    Output:
        RSS (float). 
    """   
     
    r_vec = y.reshape(N,1) - np.matmul(X, betap)
    ltwonorm_r = np.matmul(r_vec.T, r_vec)
    return 0.5*np.asscalar(ltwonorm_r)

def gradbeta(X, y, betap, N, p):
    """ Compute gradient of RSS in beta.
    Input:
        X: N x p model marix, where N is the number of observations and p the number
           of parameters (numpy array, float).
        y: N x 1 vector of responses, where N is the number of observations 
           (numpy array, float).
        betap: p x 1 vector of estimated effects (numpy array, float).
        N: number of observations (int).
        p: number of parameters or effects (int).
    Output:
        Gradient of beta (numpy array, float).
    """
    
    r_vec = y.reshape(N,1) - np.matmul(X, betap)
    grad = np.matmul(X.T, r_vec)
    return -1*grad


def AlgorithmOne(X, y, k, N, p, betam, L, tol = 0.0001, mm = 1000):
    """ Algorithm 1 in Bertsimas, D., King, A., and Mazumder, R. (2016) Best 
    subset selection via modern optimization lens. Annals of Statistics, 44, 
    813-852.
    Input:
        X: N x p model marix, where N is the number of observations and p the number
           of parameters (numpy array, float).
        y: N x 1 vector of responses, where N is the number of observations 
           (numpy array, float).
        k: number of non-zero parameters (int).
        N: number of observations (int).
        p: number of parameters or effects (int).
        betam: p x 1 vector with starting parameter estimates (numpy array, float).
        L: number of steps in the direction of the gradient (float).
        tol: tolerance between the obtained solution and the previous one (float).
        mm: maximum number of iterations for convergence (int).         
    Output:
        Starting solution and its objective function value 
        (BetaStart class).
    """
      
    diff_objval = 10**10
    cc = 0
    while (diff_objval > tol and cc < mm):
        beta_iter = np.zeros((p, 1))
        c_vec = betam - (1/L)*gradbeta(X, y, betam, N, p)
        largestkestimates = abs(c_vec).argsort(axis = 0)[-k:][::-1]
        beta_iter[largestkestimates] = c_vec[largestkestimates]
        diff_objval = gbeta(X,y,betam,N,p) - gbeta(X,y,beta_iter,N,p)
        betam = beta_iter
        cc = cc + 1
    
    # Polishing coefficients
    sel_active = np.where(betam != 0)[0] 
    Xr = X[:, sel_active]
    XsTXr = np.matmul(Xr.T,Xr)
    XrTy = np.matmul(Xr.T,y.reshape(N,1))
    
    try :
        invXsTXr = np.linalg.inv(XsTXr)
        betam[ sel_active ] = np.matmul(invXsTXr,XrTy)
    except np.linalg.linalg.LinAlgError:
        print("Alg. 1 did not polish the estimates")
    objval = gbeta(X, y, betam, N, p)
    return BetaStart(betam, objval)


def advanced_starts(X, y, k, maxiter, mymaxbeta=1000):
    """ Compute advanced starting solutions (Modified Algorithm 1)
    Input:
        X: N x p model marix, where N is the number of observations and p the number
           of parameters (numpy array, float).
        y: N x 1 vector of responses, where N is the number of observations 
           (numpy array, float).
        k: number of non-zero parameters (int).
        maxiter: number of iterations for Algorithm 1 (int).
        mymaxbeta: if no solution is found, create a synthetic solution with 
                   mymaxbeta as the maximum absolute component of that solution 
                   (float).  
    Output:
        Starting solution for MIO problem (AdvancedStart class).
    """

    # Compute number of steps in the direction of the gradient; see 
    # Bertsimas et al. (2016) for more information
    (N,p) = np.shape(X)
    XTX = np.matmul(X.T, X)
    eigenval = np.linalg.eigvals(XTX)
    L = max(eigenval)
    if L.imag != 0:
        print('Maximum eigen value is complex, results may not be accurate')
    L = L.real
    beta_results = np.zeros((p, maxiter))
    objval_vec = np.zeros((1,maxiter))
    
    # Apply Algorithm 1
    for i in range(maxiter):
        mymin = min(0, i)
        beta_init = np.random.normal(loc = mymin, scale = 2, size = p).reshape(p, 1)
        seteffzero = np.random.choice(p, size = p - k, replace = False)
        beta_init[seteffzero] = 0
        iter_result = AlgorithmOne(X, y, k, N, p, beta_init, L, tol = 0.0001, mm = 1000)
        beta_results[:,i] = iter_result.b.T
        objval_vec[:,i] = iter_result.objval      
    
    best_sol = np.argmin(objval_vec)
    startsol = beta_results[:, best_sol]
      
    # Arrange output for MIO problem
    if np.sum( abs(startsol) ) == 0 :
        print("No initial solution found")
        bp = startsol
        bm = startsol
        maxb = mymaxbeta
        z = (startsol == 0) + 0
    else :          
        bp = abs(startsol*(startsol > 0 ))
        bm = abs(startsol*(startsol < 0 ))
        maxb = max(abs(startsol))
        z = (startsol == 0) + 0
       
    return AdvancedStart(bp.tolist(), bm.tolist(), z.tolist(), maxb.tolist())

def bounds_sim(max_abs_beta, X, y, N, k, tao):
    """ Compute bounds for continuous decision variables in the MIO problem. 
    Input:
        max_abs_beta: the smallest absolute \betaˆu value known to be infeasible
                      (float).
        X: N x p model marix, where N is the number of observations and p the number
           of parameters (numpy array, float).
        y: N x 1 vector of responses, where N is the number of observations 
           (numpy array, float).
        N: number of observations (int).        
        k: number of non-zero parameters (int).
        tao: constant to safeguard against a misspecification of 'max_abs_beta' 
             (float). 
    Outputs:
        Constant values for the boosting constraints:
        - B (float).
        - BL (float).
        - E (float).
        - EL (float).        
    """
    
    B = tao*max_abs_beta
    BL = k*B

    xi_vec = []
    sumXimaxb = 0
    for i in range(N):   
        Xiabs = abs(X[i,:])
        # Get indices of the top k absolute values
        sel_maxk_subset = np.argsort( Xiabs )[-k:]
        # Sum the top k absolute values
        xi = Xiabs[sel_maxk_subset].sum() 
        xi_vec.append(xi)
        sumXimaxb += Xiabs.max()*BL

    E = max(xi_vec)*B
    yty = np.inner(y, y)
    EL = min(np.sqrt(N*yty), sumXimaxb)
    return B, E, BL, EL
    