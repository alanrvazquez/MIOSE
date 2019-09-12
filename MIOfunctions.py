#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential algorithm for MIO problems involving designs with two-level factors only, 
three-level continuous factors only and mixed-level factors.
"""
import numpy as np
from MIOproblems import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # for color bar.

# Python classess.-------------------------------------------------------------
class MIO(object):
    """
    Python class for MIO results.
    """
    
    def __init__(self, b, z, o, a, t):
        self.betas = b
        self.selected = z
        self.objvals = o
        self.gaps = a
        self.times = t

# Functions.-------------------------------------------------------------------
def sequentialAlg_twolevel(Xs, Ys, noFactors, kmax, kmin=1, nbest=1, f = 0, heredity='no', advanced=True, maxabsbeta = 10000, maxiter=100, tao=2, t=120, poolsearch = False, oflag=0):
    """ Sequential algorithm for a MIO problem involving two-level designs. 
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        noFactors: number of factors (int).
        kmax: maximum subset size for the models in the list (int).
        kmin: minimum subset size for the models in the list (int).
        nbest: number of best models to report for each k in range(kmin,kmax) (int).        
        f: maximum number of factors in a model (int). 
        heredity: type of heredity for two-factor interactions: 'no', 'weak' or 
                  'strong' (str).                 
        advanced: use advanced starts (bool).
        maxabsbeta: the smallest absolute \betaˆu value known to be infeasible
                      (float).        
        maxiter: maximum number of iterations for the modified Algorithm 1 (int).
        tao: constant to safeguard against a misspecification of 'max_abs_beta' 
             (float).         
        t: computing time limit for Gurobi (float).
        poolsearch: Use the PoolSearch implemented in gurobi to find the best
                    nbest models of size k (bool). Otherwise, use the 
                    sequential algorithm.         
        oflag: flag to show the output of gurobi: 1: show (binary).     
    
    Outputs:
        List of best fitting models: Matrix with estimated coefficients, matrix 
        with selected effects, objective function values, absolute gaps between 
        the best objective values found and their best lowerbounds, and computing 
        times required for each solution (MIO class).
        
    WARNING: The poolsearch function of gurobi, although significantly faster
             than the sequential algorith, might not give accurate solutions
             when using v7.5 and v8.0 of the solver. For details, see the file 
             MIOproblems.py.
    """
    
    (N, p) = np.shape(Xs)
    betas = np.empty((1, p))
    selectedsets = np.empty((1, p), dtype='int')
    objvals = np.empty((1, 1))
    objbounds = np.empty((1, 1))
    timesgrb = np.empty((1, 1))

    if poolsearch == True:
        print('Using Gurobi PoolSearch mode to find the best models.')

    for k in range(kmin, kmax+1):
        b, z, o, a, tt = twolevelMIOproblem(Xs, Ys, k, noFactors, nbest, f, heredity, advanced, maxabsbeta, maxiter, tao, t, poolsearch, oflag)
        betas = np.append(betas, b, axis = 0)
        selectedsets = np.append(selectedsets, z, axis = 0)
        objvals = np.append(objvals, o)
        objbounds = np.append(objbounds, a)
        timesgrb = np.append(timesgrb, tt)
    
    # Remove empty solution
    betas = np.delete(betas, 0, axis = 0)
    selectedsets = np.delete(selectedsets, 0, axis = 0)
    objvals = np.delete(objvals, 0)
    objbounds = np.delete(objbounds, 0)
    timesgrb = np.delete(timesgrb, 0)
    
    suboptimal = sum(timesgrb > t)
    if suboptimal == 0:
        print('All solutions are optimal')
    else :
        print('Number of suboptimal solutions:', suboptimal)
    
    gaps = abs(objvals - objbounds)    
    return MIO(betas, selectedsets, objvals, gaps, timesgrb) 


def sequentialAlg_threelevel(Xs, Ys, noFactors, kmax, kmin=1, nbest=1, f = 0, QE='yes', hered_int='no', QE_int_hered='no', advanced=True, maxabsbeta = 10000, maxiter=100, tao=2, t=120, poolsearch = False, oflag=0):
    """ Sequential algorithm for a MIO problem involving designs with three-level
        continuous factors only. 
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        noFactors: number of factors (int).
        kmax: maximum subset size for the models in the list (int).
        kmin: minimum subset size for the models in the list (int).
        nbest: number of best models to report for each k in range(kmin,kmax) (int).        
        f: maximum number of factors in a model (int). 
        QE: include heredity for quadratic effects (bool).
        hered_int: type of heredity for two-factor interactions: 'no', 'weak' or 
                  'strong' (str).
        QE_int_hered: type of quadratic/interaction heredity: 'no', 'weak' or 
                  'strong' (str).                                   
        advanced: use advanced starts (bool).
        maxabsbeta: the smallest absolute \betaˆu value known to be infeasible
                      (float).        
        maxiter: maximum number of iterations for the modified Algorithm 1 (int).
        tao: constant to safeguard against a misspecification of 'max_abs_beta' 
             (float).         
        t: computing time limit for Gurobi (float).
        poolsearch: Use the PoolSearch implemented in gurobi to find the best
                    nbest models of size k (bool). Otherwise, use the 
                    sequential algorithm.         
        oflag: flag to show the output of gurobi: 1: show (binary).     
    
    Outputs:
        List of best fitting models: Matrix with estimated coefficients, matrix 
        with selected effects, objective function values, absolute gaps between 
        the best objective values found and their best lowerbounds, and computing 
        times required for each solution (MIO class).
        
    WARNING: The poolsearch function of gurobi, although significantly faster
             than the sequential algorith, might not give accurate solutions
             when using v7.5 and v8.0 of the solver. For details, see the file 
             MIOproblems.py.        
    """
    
    
    (N, p) = np.shape(Xs)
    betas = np.empty((1, p))
    selectedsets = np.empty((1, p), dtype='int')
    objvals = np.empty((1, 1))
    objbounds = np.empty((1, 1))
    timesgrb = np.empty((1, 1))

    if poolsearch == True:
        print('Using Gurobi PoolSearch mode to find the best models.')

    for k in range(kmin, kmax+1):
        b, z, o, a, tt = threelevelMIOproblem(Xs, Ys, k, noFactors, nbest, f, QE, hered_int, QE_int_hered, advanced, maxabsbeta, maxiter, tao, t, poolsearch, oflag)
        betas = np.append(betas, b, axis = 0)
        selectedsets = np.append(selectedsets, z, axis = 0)
        objvals = np.append(objvals, o)
        objbounds = np.append(objbounds, a)
        timesgrb = np.append(timesgrb, tt)
    
    # Remove empty solution
    betas = np.delete(betas, 0, axis = 0)
    selectedsets = np.delete(selectedsets, 0, axis = 0)
    objvals = np.delete(objvals, 0)
    objbounds = np.delete(objbounds, 0)
    timesgrb = np.delete(timesgrb, 0)
    
    suboptimal = sum(timesgrb > t)
    if suboptimal == 0:
        print('All solutions are optimal')
    else :
        print('Number of suboptimal solutions:', suboptimal)
    
    gaps = abs(objvals - objbounds)    
    return MIO(betas, selectedsets, objvals, gaps, timesgrb) 

def sequentialAlg_mixedlevel(Xs, Ys, notwolevelFactors, kmax, kmin=1, nbest=1, group = True, mix_eff_indices = 0, hered_int='no', advanced=True, maxabsbeta = 10000, maxiter=100, tao=2, t=120, poolsearch = False, oflag=0):
    """ Sequential algorithm for a MIO problem involving designs with two-level
        factors and multi-level categorical factors. 
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        noFactors: number of factors (int).
        kmax: maximum subset size for the models in the list (int).
        kmin: minimum subset size for the models in the list (int).
        nbest: number of best models to report for each k in range(kmin,kmax) (int).        
        group: include grouping constraints for multi-level categorical factors
               (bool).
        mix_eff_indices: n x (s-1) matrix with the indices of the effects of the 
                         multi-level categorical factors in the model matrix, where
                         n is the number of categorical factors and s the number
                         of levels (numpy array, int).
        hered_int: type of heredity for two-factor interactions: 'no', 'weak' or 
                  'strong' (str).                                  
        advanced: use advanced starts (bool).
        maxabsbeta: the smallest absolute \betaˆu value known to be infeasible
                      (float).        
        maxiter: maximum number of iterations for the modified Algorithm 1 (int).
        tao: constant to safeguard against a misspecification of 'max_abs_beta' 
             (float).         
        t: computing time limit for Gurobi (float).
        poolsearch: Use the PoolSearch implemented in gurobi to find the best
                    nbest models of size k (bool). Otherwise, use the 
                    sequential algorithm.         
        oflag: flag to show the output of gurobi: 1: show (binary).     
    
    Outputs:
        List of best fitting models: Matrix with estimated coefficients, matrix 
        with selected effects, objective function values, absolute gaps between 
        the best objective values found and their best lowerbounds, and computing 
        times required for each solution (MIO class).

    WARNING: The poolsearch function of gurobi, although significantly faster
             than the sequential algorith, might not give accurate solutions
             when using v7.5 and v8.0 of the solver. For details, see the file 
             MIOproblems.py.        
    """
    
    
    (N, p) = np.shape(Xs)
    betas = np.empty((1, p))
    selectedsets = np.empty((1, p), dtype='int')
    objvals = np.empty((1, 1))
    objbounds = np.empty((1, 1))
    timesgrb = np.empty((1, 1))

    if poolsearch == True:
        print('Using Gurobi PoolSearch mode to find the best models.')
    
    for k in range(kmin, kmax+1):
        b, z, o, a, tt = mixedlevelMIOproblem(Xs, Ys, k, notwolevelFactors, nbest, group, mix_eff_indices, hered_int, advanced, maxabsbeta, maxiter, tao, t, poolsearch, oflag)
        betas = np.append(betas, b, axis = 0)
        selectedsets = np.append(selectedsets, z, axis = 0)
        objvals = np.append(objvals, o)
        objbounds = np.append(objbounds, a)
        timesgrb = np.append(timesgrb, tt)
    
    # Remove empty solution
    betas = np.delete(betas, 0, axis = 0)
    selectedsets = np.delete(selectedsets, 0, axis = 0)
    objvals = np.delete(objvals, 0)
    objbounds = np.delete(objbounds, 0)
    timesgrb = np.delete(timesgrb, 0)
    
    suboptimal = sum(timesgrb > t)
    if suboptimal == 0:
        print('All solutions are optimal')
    else :
        print('Number of suboptimal solutions:', suboptimal)
    
    gaps = abs(objvals - objbounds)        
    return MIO(betas, selectedsets, objvals, gaps, timesgrb) 

def raster_plot(mio_results, labels, color = True, plttimes = False, nfreqlab = 5, width = 14, height = 10):
    """ Create raster plot of the (unique) best-fitting models found.
    Inputs:
        mio_result: list of models (MIO class)
        labels: labels for the effects (list, str)
        color: display the raster plot in color (bool).
        plttimes: plot the computing times for each solution (bool).
        nfreqlab: show the effects which appear more than 'nfreqlab' times in 
                  the models (int).
        width: width of the raster plot (float).
        height: height of the raster plot (float).
    Outputs:
        plt: raster plot for the list of models and a line plot for the computing
        times of each solution. 
    """
    
    nonzero_estimates = np.where(np.sum(mio_results.selected, axis = 0) > nfreqlab)[0]
    sel_labels = np.take(labels, nonzero_estimates)
    
    # Set figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = width
    fig_size[1] = height
    plt.rcParams["figure.figsize"] = fig_size
                
    # Remove repeated models.
    nmod = np.shape(mio_results.selected)[0]
    u, indices = np.unique(mio_results.selected,axis=0,return_index=True)
    indun = np.sort(indices) # indices of unique rows.
    betas = mio_results.betas[indun,:]
    RSS = mio_results.objvals[indun]
    nrepeat = nmod - np.shape(u)[0]
    if nrepeat > 0:
        print(str(nrepeat)+' repeated models were removed from the raster plot.')                
    myorder = RSS.argsort()
    M = betas[myorder]
    # Raster plot
    plt.figure(1)
    cmap = plt.cm.Greys
    mabsebeta = np.max(abs(M))
    plt.gca().invert_yaxis()
    if color :
        im = plt.imshow(M, cmap = 'bwr', aspect = 'auto', vmin = -1*mabsebeta, vmax = mabsebeta)
    else: 
        im = plt.imshow(abs(M), cmap=cmap, aspect = 'auto', vmin = 0, vmax = mabsebeta)
    plt.xlabel('Effect')
    plt.ylabel('Ranking of models')
    plt.xticks(nonzero_estimates, sel_labels)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im,cax=cax)
    plt.show()
    ax.invert_yaxis()
    
    if plttimes :
        # Computing time of solutions
        plt.figure(2)
        nmodels = np.shape(mio_results.betas)[0]
        plt.plot(range(nmodels), mio_results.times)
        plt.xlabel('Solution')
        plt.ylabel('Time (s)')
        
    return plt
