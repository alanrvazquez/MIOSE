#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 4: Router bit experiment
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import time
import sklearn.preprocessing # to scale matrices
from sklearn import linear_model

from MIOfunctions import * # import MIO functions

#%% Read data.-----------------------------------------------------------------

# Read data in excel, remove header
mydata = np.genfromtxt('data/Router.csv', delimiter = ',', skip_header = 1)

y = mydata[:, 11] # response vector
twoD = mydata[:, [0, 1, 2, 7, 8, 9, 10]] # matrix with two-level factors
n_twolevel = np.shape(twoD)[1] # number of two-level factors
fourD = mydata[:, 3:7] # matrix with four-level factors
hered_int = 'strong' # type of constraints for two-factor interactions among two-level factors
group = True # grouping constraints
kmax = 14
nbest = 10

# Save best models
filename = 'router_kmax' + str(kmax) + '_nbest_' + str(nbest) + '_SeqAlg.pkl'
#%% Manually generate model matrix and select response vector.-----------------
# Model matrix with two-factor interactions for the two-level factors
twomodmat = twofimodmat(twoD)
# Contrast vectors for four-level factor effects
cvfourlevel = np.concatenate((fourD[:,:2], np.reshape(fourD[:,0]*fourD[:,1], (32,1)), fourD[:,2:], 
                              np.reshape(fourD[:,2]*fourD[:,3], (32,1))), axis = 1)

# Model matrix 
# Warning: Four-level factor effects must be at the end of the matrix.
Xs = np.concatenate((twomodmat.x, cvfourlevel), axis=1)
# Indices of four-factor effects
mix_eff = np.array([[28,29, 30], [31, 32, 33]])           
# Centered response vector
Ys = y - np.mean(y) 
Ys.transpose()

#%% Fit main effects model.----------------------------------------------------
regr = linear_model.LinearRegression()
# Compute model including linear and quadratic main effects
regr.fit(mydata[:, :11], y)
# Maximum feasible effect 
maxabsbeta = max(abs(regr.coef_))   

#%% Create list of models. ----------------------------------------------------
start_time = time.time()
list_of_models = sequentialAlg_mixedlevel(Xs=Xs,Ys=y,notwolevelFactors=n_twolevel,
                                                    kmax=kmax, nbest=nbest,group=group, 
                                                    mix_eff_indices = mix_eff,
                                                    hered_int=hered_int,
                                                    maxabsbeta=maxabsbeta,advanced=False)
mytime = time.time() - start_time

# Save best models
save_object(list_of_models, filename)         
print('Computing time (s):' + str(mytime))

#%% Create plots.--------------------------------------------------------------
# Command to load files
mio_results = pickle.load( open( filename, "rb" ) )
# Manually create labels for model matrix
# Labels for the MEs
factors = ['A', 'B', 'C', 'F', 'G', 'H', 'I']
mylabels = factors
# Construct quadratic model matrix
for i, j in itertools.combinations(range(7),2):
    # Labels for 2FIs
    lab_int = ':'.join([factors[i], factors[j]])
    mylabels.append(lab_int)         

mylabels.extend(['Da', 'Db', 'Da:Db', 'Ea', 'Eb', 'Ea:Eb'])

# Create raster plot
raster_plot(mio_results, mylabels)


