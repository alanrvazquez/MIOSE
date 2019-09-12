#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIO problems for designs 
    - with two-level factors only, 
    - with three-level continuous factors only,     
    - with two-level and three-level factors.    
"""

# Import libraries
from gurobipy import *
import numpy as np
from itertools import *
from FOalgorithms import *
from matfunctions import *
  
#Auxiliary functions.------------------------------------------------------------
def tolerance_integer(a, tol = 0.0001):
    """ Transform a numeric value to a binary variable.
        This function corrects for possible misspecifications of the values
        of the binary decision variables resulting from the Gurobi Optimization.
    Input:
        a: variable value (int or float).
        tol: tolerance to be considered as integer (float).
    Output:
        zint: the integer value of a (int).
    """
    if abs(a - 1) < tol:
        zint = 1
    elif abs(a - 0) < tol:
        zint = 0
    else :
        zint = a
        print('Decision variable z did not converge to an integer, ' + str(a))
    return(zint)    


def get_solution(model, p):
    """ Obtain the solutions from a gurobi model.
    Input:
        model: gurobi model.
        p: number of parameters of effects (int).
    Output:
        beta: 1 x p vector of estimated effects, where p is the number of parameters
             (numpy array, float).
        z: 1 x p vector of selected effects (numpy array, int).
        best_subset: list of indices of the selected effects (list, int).
    """
    beta = np.zeros((1, p))
    z = np.zeros((1, p), dtype = 'int')
    best_subset = []
    for v in range(p):
        a = model.getVarByName('a_'+ str(v))
        bp = model.getVarByName("b_"+ str(v) +"_0")
        bn = model.getVarByName("b_"+ str(v) +"_1")
        beta[0,v] = bp.x - bn.x
        aint = tolerance_integer(a.x)
        z[0, v] = 1 - aint    
        if (1 - aint) > 0.5:
            best_subset.append(v)
    return beta, z, best_subset

def get_subopt_solution(model, p):
    """ Obtain the sub-optimal solutions from a gurobi model. This fuction reports
        the solutions obtained using gurobi poolsearch.
    Input:
        model: gurobi model.
        p: number of parameters of effects (int).
    Output:
        beta: 1 x p vector of estimated effects, where p is the number of parameters
             (numpy array, float).
        z: 1 x p vector of selected effects (numpy array, int).
        best_subset: list of indices of the selected effects (list, int).
    """
    beta = np.zeros((1, p))
    z = np.zeros((1, p), dtype = 'int')
    best_subset = []
    for v in range(p):
        a = model.getVarByName('a_'+ str(v))
        bp = model.getVarByName("b_"+ str(v) +"_0")
        bn = model.getVarByName("b_"+ str(v) +"_1")
        beta[0,v] = bp.xn - bn.xn
        aint = tolerance_integer(a.xn)
        z[0, v] = 1 - aint    
        if (1 - aint) > 0.5:
            best_subset.append(v)
    return beta, z, best_subset

#List of MIO problems.---------------------------------------------------------
def twolevelMIOproblem(Xs, Ys, k, noFactors, nbest = 1, f = 0, heredity = 'no', advanced = True, maxabsbeta = 1000, maxiter = 100, tao = 2, t = 120, poolsearch = False, oflag = 1):
    """ MIO problem for two-level designs. The problem can include the heredity
        constraints for the two-factor interactions.
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        k: maximum number of non-zero parameters (int).
        noFactors: number of factors (int).
        nbest: number of best models to report (int).
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
        betaMat: nbest x p matrix with estimated coefficients (numpy array, 
                 float).
        zMat: nbest x p matrix with the selected variables for each optimal 
              solution (numpy array, int).
        objvalMat: 1 x nbest vector with the objective values for the optimal 
              solutions (numpy array, float).
        objboundMat: 1 x nbest vector with the best known bounds on the objective 
              values (numpy array, float).      
        timeMat: 1 x nbest vector with the computing times for the optimal 
              solutions (numpy array, float).
              
    WARNING: The pool search mode of gurobi is significantly faster than the
             sequential algorithm. However, the poolsearch functionality in gurobi 
             v7.5 and v8.0 does not work well for mixed-integer quadratic problems 
             as this one. The function does not report the correct objective function 
             values and it also repeats solutions in the pool of best solutions. 
             The developers of gurobi said that they will fix this issue in 
             future versions of the solver. For more information, see the following
             website: 
             https://support.gurobi.com/hc/en-us/community/posts/360043638232-PoolObjVal-issues  
    """
  
    try: 
        #==LOAD DATA===================================================
        
        model_name = "MIO_problem_twolevel"
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
        
    
        #==COMPUTE BOUNDS======================================================      
        if advanced:
            # Compute starting solution
            startsol = advanced_starts(Xs, Ys, k, maxiter)
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(startsol.maxb, Xs, Ys, n, k, tao)
        else :
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(maxabsbeta, Xs, Ys, n, k, tao)
        
        #==CREATE VARIABLES====================================================
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

            model.addConstr(quicksum( w_vecvar[i] for i in range(noFactors)) <= f, name="factor_spar") 

        # Heredity constraints=================================================
        if heredity == 'strong':
            nME_choose_two = itertools.combinations(range(noFactors), 2)
            a_ij = noFactors
            for i,j in nME_choose_two: 
                model.addConstr( a_vecvar[a_ij] - a_vecvar[i] >= 0, name = "heredity"+str(i) )
                model.addConstr( a_vecvar[a_ij] - a_vecvar[j] >= 0, name = "heredity"+str(j) )
                a_ij += 1                          
    
        if heredity == 'weak':     
            nME_choose_two = itertools.combinations(range(noFactors), 2)
            a_ij = noFactors
            for i,j in nME_choose_two: 
                model.addConstr( 1 + a_vecvar[a_ij] - a_vecvar[i] - a_vecvar[j] >= 0, name = "heredity"+str(i)+str(j) )
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
            timeMat[0,0] = model.Runtime
            objboundMat[0,0] = model.objBound
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
    
       
    except GurobiError:
        error = 1
        error_type = 'Encountered a Gurobi error'
        print('Encountered a Gurobi error')
    
    except AttributeError:
        error = 1
        error_type = 'Encountered an attribute error'
        print('Encountered an attribute error')
        
    return betaMat, zMat, objvalMat, objboundMat, timeMat    
    
def threelevelMIOproblem(Xs, Ys, k, noFactors, nbest = 1, f = 0, QE = True, hered_int = 'no', QE_int_hered = 'no', advanced = True, maxabsbeta = 1000, maxiter = 100, tao = 2, t = 120, poolsearch = False, oflag = 1):
    """ MIO problem for designs with three-level continuous factors. The problem 
        can include the heredity constraints for the second-order effects, in the form
        of two-factor interactions and quadratic effects.
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        k: maximum number of non-zero parameters (int).
        noFactors: number of factors (int).
        nbest: number of best models to report (int).
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
        betaMat: nbest x p matrix with estimated coefficients (numpy array, 
                 float).
        zMat: nbest x p matrix with the selected variables for each optimal 
              solution (numpy array, int).
        objvalMat: 1 x nbest vector with the objective values for the optimal 
              solutions (numpy array, float).
        objboundMat: 1 x nbest vector with the best known bounds on the objective 
              values (numpy array, float).               
        timeMat: 1 x nbest vector with the computing times for the optimal 
              solutions (numpy array, float).
              
    WARNING: The pool search mode of gurobi is significantly faster than the
             sequential algorithm. However, the poolsearch functionality in gurobi 
             v7.5 and v8.0 does not work well for mixed-integer quadratic problems 
             as this one. The function does not report the correct objective function 
             values and it also repeats solutions in the pool of best solutions. 
             The developers of gurobi said that they will fix this issue in 
             future versions of the solver. For more information, see the following
             website: 
             https://support.gurobi.com/hc/en-us/community/posts/360043638232-PoolObjVal-issues  
    """
  
    try: 
        #==LOAD DATA===========================================================
        
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
    
        #==COMPUTE BOUNDS======================================================      
        if advanced:
            # Compute starting solution      
            startsol = advanced_starts(Xs, Ys, k, maxiter)
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(startsol.maxb, Xs, Ys, n, k, tao)
        else :
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(maxabsbeta, Xs, Ys, n, k, tao)
        
        #==CREATE VARIABLES====================================================
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
    
       
    except GurobiError:
        error = 1
        error_type = 'Encountered a Gurobi error'
        print('Encountered a Gurobi error')
    
    except AttributeError:
        error = 1
        error_type = 'Encountered an attribute error'
        print('Encountered an attribute error')
        
    return betaMat, zMat, objvalMat, objboundMat, timeMat    



def mixedlevelMIOproblem(Xs, Ys, k, notwolevelFactors, nbest = 1, group = False, mix_eff_indices = 0, hered_int = 'no', advanced = True, maxabsbeta = 1000, maxiter = 100, tao = 2, t = 120, poolsearch = False, oflag = 1):
    """ MIO problem for designs with two-level factors and multi-level categorical 
    factors. The problem can include the heredity constraints for the two-factor 
    interactions between the two-level factors only.
    Inputs:
        Xs: N x p scaled model marix, where N is the number of observations and p 
            the number of parameters (numpy array, float).
        Ys: N x 1 vector of centered responses (numpy array, float).             
        k: maximum number of non-zero parameters (int).
        notwolevelFactors: number of two-level factors (int).
        nbest: number of best models to report (int).
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
        betaMat: nbest x p matrix with estimated coefficients (numpy array, 
                 float).
        zMat: nbest x p matrix with the selected variables for each optimal 
              solution (numpy array, int).
        objvalMat: 1 x nbest vector with the objective values for the optimal 
              solutions (numpy array, float).
        objboundMat: 1 x nbest vector with the best known bounds on the objective 
              values (numpy array, float).               
        timeMat: 1 x nbest vector with the computing times for the optimal 
              solutions (numpy array, float).

    WARNING: The pool search mode of gurobi is significantly faster than the
             sequential algorithm. However, the poolsearch functionality in gurobi 
             v7.5 and v8.0 does not work well for mixed-integer quadratic problems 
             as this one. The function does not report the correct objective function 
             values and it also repeats solutions in the pool of best solutions. 
             The developers of gurobi said that they will fix this issue in 
             future versions of the solver. For more information, see the following
             website: 
             https://support.gurobi.com/hc/en-us/community/posts/360043638232-PoolObjVal-issues  
    """
  
    try: 
        #==LOAD DATA===================================================
        
        model_name = "MIO_problem_mixedlevel"
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
    
        #==COMPUTE BOUNDS==============================================      
        if advanced:
            # Compute starting solution      
            startsol = advanced_starts(Xs, Ys, k, maxiter)               
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(startsol.maxb, Xs, Ys, n, k, tao)
        else :
            max_betaval, max_etaval, maxlone_normbeta, maxlone_normeta = bounds_sim(maxabsbeta, Xs, Ys, n, k, tao)
        
        #==CREATE VARIABLES====================================================
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
                                                          
        # Heredity constraints=================================================
        if hered_int == 'strong':
            nME_choose_two = itertools.combinations(range(notwolevelFactors), 2)
            a_ij = notwolevelFactors
            for i,j in nME_choose_two: 
                model.addConstr( a_vecvar[a_ij] - a_vecvar[i] >= 0, name = "heredity"+str(i) )
                model.addConstr( a_vecvar[a_ij] - a_vecvar[j] >= 0, name = "heredity"+str(j) )
                a_ij += 1                          
    
        if hered_int == 'weak':     
            nME_choose_two = itertools.combinations(range(notwolevelFactors), 2)
            a_ij = notwolevelFactors
            for i,j in nME_choose_two: 
                model.addConstr( 1 + a_vecvar[a_ij] - a_vecvar[i] - a_vecvar[j] >= 0, name = "heredity"+str(i)+str(j) )
                a_ij += 1          

        # Grouping constraints================================================= 
        if group:
           for i in range(np.shape(mix_eff_indices)[0]):
               for j in range(1, np.shape(mix_eff_indices)[1]):
                   model.addConstr( a_vecvar[mix_eff_indices[i,0]] == a_vecvar[mix_eff_indices[i,j]], name = "grouping"+str(i))
             
        #==PROVIDE ADVANCED STARTS=============================================
        if advanced:
            for i in range(p):
                a_vecvar[i].start = startsol.z[i]        
                b_matvar[i][0].start = startsol.bp[i]
                b_matvar[i][1].start = startsol.bm[i]
    
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
            timeMat[0,0] = model.Runtime
            objboundMat[0,0] = model.objBound       
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
    
       
    except GurobiError:
        error = 1
        error_type = 'Encountered a Gurobi error'
        print('Encountered a Gurobi error')
    
    except AttributeError:
        error = 1
        error_type = 'Encountered an attribute error'
        print('Encountered an attribute error')
        
    return betaMat, zMat, objvalMat, objboundMat, timeMat    
    
    