#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:38:19 2023

@author: alanvazquez
"""

#%%
Ys = y 
kmin=1
nbest=1
f = 0
QE='yes'
hered_int='no'
QE_int_hered='no'
advanced=True
maxabsbeta = 10000
maxiter=100
tao=2
t=120
poolsearch = False
oflag=0
#%%       

(N, p) = np.shape(Xs)
betas = np.empty((1, p))
selectedsets = np.empty((1, p), dtype='int')
objvals = np.empty((1, 1))
objbounds = np.empty((1, 1))
timesgrb = np.empty((1, 1))

if poolsearch == True:
    print('Using Gurobi PoolSearch mode to find the best models.')

#%%
for k in range(kmin, kmax+1):
    b, z, o, a, tt = threelevelMIOproblem(Xs, Ys, k, noFactors, nbest, f, QE, hered_int, QE_int_hered, advanced, maxabsbeta, maxiter, tao, t, poolsearch, oflag)
    betas = np.append(betas, b, axis = 0)
    selectedsets = np.append(selectedsets, z, axis = 0)
    objvals = np.append(objvals, o)
    objbounds = np.append(objbounds, a)
    timesgrb = np.append(timesgrb, tt)

#%%
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

#%% MIO Problem
k = 5 
nbest = 1
f = 0
QE = True
hered_int = 'no'
QE_int_hered = 'no' 
advanced = True 
maxabsbeta = 1000 
maxiter = 100 
tao = 2 
t = 120 
poolsearch = False 
oflag = 1

#%%==LOAD DATA===========================================================

model_name = "MIO_problem_threelevel"
error = 0   
# Design information
(n, p) = np.shape(Xs) # Run size
 
# Create model 
model = Model(model_name)
# Set parameters in the model
model.params.outputflag = oflag
if poolsearch == True:
    model.params.poolsearchmode = 2 # Controls the approach to find solutions.
    model.params.poolsolutions = nbest # Target number of solutions to find.        

#%%==COMPUTE BOUNDS======================================================      
if advanced:
    # Compute starting solution      
    startsol = advanced_starts(Xs, Ys, k, maxiter)
    max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(startsol.maxb, Xs, Ys, n, k, tao)
else :
    max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(maxabsbeta, Xs, Ys, n, k, tao)

#%%==CREATE VARIABLES====================================================
# Binary variables
a_vecvar = []
for i in range(p):
    def_bin_var = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="a_{0}".format(i))
    a_vecvar.append(def_bin_var)
    
b_matvar = []
for i in range(p):
    row = []
    for j in range(2):
        def_cont_var = model.addVar(lb=0,ub=max_betaval,vtype=GRB.CONTINUOUS, name = "b_{0}_{1}".format(i,j))
        row.append(def_cont_var)
    b_matvar.append(row)

e_matvar = []
for i in range(n):
    row = []
    for j in range(2):
        def_cont_var = model.addVar(lb=0,ub=max_etaval,vtype=GRB.CONTINUOUS, name = "e_{0}_{1}".format(i,j))
        row.append(def_cont_var)
    e_matvar.append(row)

if f > 0: # If factor sparsity
    w_vecvar = []
    for i in range(noFactors):
        def_bin_var = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="w_{0}".format(i))
        w_vecvar.append(def_bin_var)

# Integrate new variables
model.update()

#==SET OBJECTIVE FUNCTION==============================================
yty = np.inner(Ys, Ys)
Xty = np.dot(Xs.T,Ys)
Xtyb_linexp = quicksum(Xty[i]*(b_matvar[i][0] - b_matvar[i][1]) for i in range(p))
etateta_quadexp = quicksum(e_matvar[i][0]*e_matvar[i][0] + e_matvar[i][1]*e_matvar[i][1] - 2*e_matvar[i][0]*e_matvar[i][1]
                          for i in range(n))
model.setObjective(0.5*etateta_quadexp - Xtyb_linexp + 0.5*yty, GRB.MINIMIZE)

#==DEFINE AND ADD CONSTRAINTS==========================================
# Sparsity constraint sum(z) <= k
model.addConstr(quicksum( (1 - a_vecvar[i]) for i in range(p)) <= k, name="spar")

# Constraint of the form eta = Xsbeta 
for c in range(n):
    model.addConstr( quicksum( Xs[c,i]*(b_matvar[i][0] - b_matvar[i][1])
                                       for i in range(p) ) - e_matvar[c][0] + e_matvar[c][1] == 0,
                     name = "etaXbeta_const" + str(c))

# Add SOS constraints for beta variables
for s in range(p):
    model.addSOS(GRB.SOS_TYPE1, [b_matvar[s][0], b_matvar[s][1], a_vecvar[s]])
    
# L1 norm of beta and eta are bounded
model.addConstr(quicksum(b_matvar[i][0] + b_matvar[i][1] for i in range(p)) <= maxlone_normbeta,
                name = "L1norm_beta")
model.addConstr(quicksum(e_matvar[i][0] + e_matvar[i][1] for i in range(n)) <= maxlone_normeta,
                name = "L1norm_eta")  
                                                  
# Factor sparsity======================================================
if f > 0:
    for i in range(noFactors):
        model.addConstr( w_vecvar[i] + a_vecvar[i] - 1 >= 0, name = "factor_spar_me"+str(i) )

    # If there are interactions
    if p > noFactors: 
        nME_choose_two = itertools.combinations(range(noFactors), 2)
        a_ij = noFactors
        for i,j in nME_choose_two: 
            model.addConstr( w_vecvar[i] + a_vecvar[a_ij] - 1 >= 0, name = "factor_spar_int"+str(i) )
            model.addConstr( w_vecvar[j] + a_vecvar[a_ij] - 1 >= 0, name = "factor_spar_int"+str(j) )
            a_ij += 1 
    
    # If there are quadratic effects
    a_ij = noFactors + nchoosek(noFactors,2)
    if p > a_ij: 
        seq_nME = range(noFactors)
        for i in seq_nME: 
            model.addConstr( w_vecvar[i] + a_vecvar[a_ij] - 1 >= 0, name = "factor_spart_quad"+str(i) )
            a_ij += 1

    model.addConstr(quicksum( w_vecvar[i] for i in range(noFactors)) <= f, name="factor_spar") 

# Heredity constraints=================================================
if hered_int == 'strong':
    nME_choose_two = itertools.combinations(range(noFactors), 2)
    a_ij = noFactors
    for i,j in nME_choose_two: 
        model.addConstr( a_vecvar[a_ij] - a_vecvar[i] >= 0, name = "heredity"+str(i) )
        model.addConstr( a_vecvar[a_ij] - a_vecvar[j] >= 0, name = "heredity"+str(j) )
        a_ij += 1                          

if hered_int == 'weak':     
    nME_choose_two = itertools.combinations(range(noFactors), 2)
    a_ij = noFactors
    for i,j in nME_choose_two: 
        model.addConstr( 1 + a_vecvar[a_ij] - a_vecvar[i] - a_vecvar[j] >= 0, name = "heredity"+str(i)+str(j) )
        a_ij += 1          

#Constraints for quadratic effects=====================================
if QE:     
    seq_nME = range(noFactors)
    a_ij = noFactors + nchoosek(noFactors,2)
    for i in seq_nME: 
        model.addConstr( a_vecvar[a_ij] - a_vecvar[i] >= 0, name = "quadratic"+str(i) )
        a_ij += 1

# Quadratic/interaction heredity=======================================
if QE_int_hered == 'strong':
    nME_choose_two = itertools.combinations(range(noFactors), 2)
    a_ij = noFactors        
    start_quadratic = noFactors + nchoosek(noFactors,2)
    for i,j in nME_choose_two:
        z = i + start_quadratic
        w = j + start_quadratic            
        model.addConstr( a_vecvar[a_ij] - a_vecvar[z] >= 0, name = "qe_int_heredity"+str(z) )
        model.addConstr( a_vecvar[a_ij] - a_vecvar[w] >= 0, name = "qe_int_heredity"+str(w) )
        a_ij += 1
        
if QE_int_hered == 'weak':
    nME_choose_two = itertools.combinations(range(noFactors), 2)
    a_ij = noFactors        
    start_quadratic = noFactors + nchoosek(noFactors,2)
    for i,j in nME_choose_two:
        z = i + start_quadratic
        w = j + start_quadratic            
        model.addConstr( 1 + a_vecvar[a_ij] - a_vecvar[z] - a_vecvar[w] >= 0, name = "qe_int_heredity"+str(z)+str(w) )
        a_ij += 1 
            
#==SET TIME LIMIT====================================================== 
model.params.timeLimit = t     

#==OPTIMIZE MODEL======================================================
model.optimize()

#==GET SOLUTION =======================================================
beta, z, best_subset = get_solution(model, p)

#==GENERATE LIST OF MODELS=============================================
if poolsearch == True:
    nfound = model.solcount
    betaMat = np.zeros((nfound, p))
    zMat = np.zeros((nfound, p), dtype = 'int')
    objvalMat = np.zeros((1, nfound))
    objboundMat = np.zeros((1, nbest))
    # Assign optimal solutions
    betaMat[0,:] = beta
    zMat[0,:] = z   
    objvalMat[0,0] = fit_model_rss(Xs, Ys, z[0,:])
    timeMat = model.Runtime
    
    for i in range(1, nfound):
        # Retrieve sub-optimal solutions from the pool of best solutions.
        model.params.solutionnumber = i
        beta, z, best_subset = get_subopt_solution(model, p)
        betaMat[i,:] = beta
        zMat[i,:] = z
        objvalMat[0,i] = fit_model_rss(Xs, Ys, z[0,:])
        
else: 
    betaMat = np.zeros((nbest, p))
    zMat = np.zeros((nbest, p), dtype = 'int')
    objvalMat = np.zeros((1, nbest))
    objboundMat = np.zeros((1, nbest))
    timeMat = np.zeros((1, nbest))        
    # Assign optimal solutions
    betaMat[0,:] = beta
    zMat[0,:] = z   
    objvalMat[0,0] = model.objVal
    objboundMat[0,0] = model.objBound
    timeMat[0,0] = model.Runtime
    
    for i in range(1, nbest):
        # Block optimal model
        model.addConstr(quicksum((1 - a_vecvar[i]) for i in best_subset) <= k-1, name = "block_model"+str(i))
        # Solve new MIO problem
        model.optimize()
        beta, z, best_subset = get_solution(model, p)
        betaMat[i,:] = beta
        zMat[i,:] = z
        objvalMat[0,i] = model.objVal
        objboundMat[0,i] = model.objBound         
        timeMat[0,i] = model.Runtime       

       
  
