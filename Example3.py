#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3: TNO mirror image polishing experiment
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
from MIOfunctions import *
import time
import pickle

#%% Read data.-----------------------------------------------------------------

# Read data in excel, remove header
noFactors = 13
mydata = np.genfromtxt('data/TNO.csv', delimiter = ',', skip_header = 1)
D = mydata[:,0:noFactors] # select design
y = mydata[:,noFactors] # select responses
nbest = 10 # number of best models
kmax = 11 # maximum model size
myhered = 'weak' # heredity type: 'weak' or 'strong'

# Data preprocessing
y = y - np.mean(y) # centered response vector
y.transpose()
# Create two-factor interaction matrix
X = twofimodmat(D)

# Save best models
filename = 'tno_kmax' + str(kmax) + '_nbest_' + str(nbest) + '_SeqAlg.pkl'
#%% Create list of models.-----------------------------------------------------
start_time = time.time()
list_of_models = sequentialAlg_twolevel(Xs=X.x,Ys=y,noFactors=noFactors, kmax=kmax,  
                                       nbest=nbest,heredity=myhered, t=200)

mytime = time.time() - start_time
                  
# Save best models
save_object(list_of_models, filename)
print('Computing time (s):' + str(mytime))

#%% Load and report results.---------------------------------------------------
# Command to load files
mio_results = pickle.load( open( filename, "rb" ) )

# Create raster plots
raster_plot(mio_results, X.labels)

