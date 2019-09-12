#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: Simulated example involving a 10-factor 21-run definitive screening 
           design.
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn.preprocessing # to scale matrices
import pickle
import time

from MIOfunctions import * # import MIO functions

#%% Read data.-----------------------------------------------------------------

# Read data in excel, remove header
noFactors = 10
mydata = np.genfromtxt('data/Simulated.csv', delimiter = ',', skip_header = 1)
D = mydata[:,0:noFactors] # select design
y = mydata[:,noFactors] # select responses
kmax = 10 # maximum model size
nbest = 10 # number of models for each model size

QE = True # heredity constraints for quadratic effects
hered_int = 'weak' # type of heredity for two-factor interactions
QE_int_hered = 'weak' # type of quadratic/interaction heredity

# Data preprocessing
y = y - np.mean(y) # centered response vector
y.transpose() 
X = quadmodmat(D) # compute full quadratic model matrix
Xs = sklearn.preprocessing.scale(X.x) # scale matrix            

# Save best models
filename = 'simulated_kmax' + str(kmax) + '_nbest_' + str(nbest) + '_SeqAlg.pkl'
                             
#%% Create list of models.-----------------------------------------------------
start_time = time.time()
list_of_models = sequentialAlg_threelevel(Xs=Xs, Ys=y, noFactors=noFactors, 
                                          kmax=kmax, nbest=nbest, QE=QE, 
                                          hered_int=hered_int, 
                                          QE_int_hered=QE_int_hered, t=200)
mytime = time.time() - start_time

# Save best models
save_object(list_of_models, filename)         
print('Computing time (s):' + str(mytime))

#%% Create plots.--------------------------------------------------------------

# Command to load files
mio_results = pickle.load( open( filename, "rb" ) )
# Create raster plot
raster_plot(mio_results, X.labels)

