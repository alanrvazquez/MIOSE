README file for code to accompany "A Mixed Integer Optimization Approach 
for Model Selection in Screening Experiments." by Alan R. Vazquez,
Eric D. Schoen and Peter Goos.

===CONTENTS=======================================

A. SCRIPT FILES
B. PRIMARY FILES
C. UTILITY FILES
D. DATA FILES
E. INSTALLATION

==================================================

A) SCRIPT FILES.
Use these files for data analysis using MIO.

Example1.py       			Shows how to analyze the data from a three-level screening 
							design subject to heredity and quadratic/interaction 
							heredity constraints. It uses the simulated experiment in 
							the main text as an example.

Example2.py       			Shows how to analyze the data from a three-level screening 
							design subject to heredity constraints. It involves a synthetic
							experiment in which the active effects do not follow effect
							heredity.							
                 
Example3.py    				Shows how to analyze the data from a two-level screening 
							design subject to heredity constraints. It uses the diamond 
							polishing experiment in the main text as an example.

Example4.py    				Shows how to analyze the data from a mixed-level screening 
							design (two- and four-level factors) subject to heredity 
							constraints for the two-level factors. It uses the router bit 
							experiment in the main text as an example.

B) PRIMARY FILES.
These files contain the functions one would interact with when analyzing data with MIO. 

MIOfunctions.py  			Functions to generate the list of best-fitting models (Algorithm 1 
							in the main text). There is a function for each case of analysis of
							a two-, three- and mixed-level design.                     

C) UTILITY FILES.
These files contain functions for specific purposes.

MIOproblems.py     			Gurobi/Python implementations of the MIO problems. There is one 
							MIO problem each case of analysis of a two-, three- and mixed-level 
							design.     

FOAlgorithms.py           	Routines to generate an advanced starting solution to the MIO problem.
							This file contains the implementation of Algorithm 1 in Bertsimas, D.,
							King, A. and Mazumder, R. (2016) Bestsubset selection via modern 
							optimization lens. Annals of Statistics, 44, 813-852.
                        
matfunctions.m             	Create two-factor interaction and full quadratic model matrices among 
							other python functions.
                        
D) DATA FILES.

Simulated.csv         		Data for the simulated experiments, Examples 1 and 2 in the main text.

TNO.csv         			Data for the diamond polishing experiment, Example 3 in the main text.

Router.csv         			Data for the router bit experiment, Example 4 in the main text.

E) INSTALLATION. 
Please follow these steps to run the routines:

1) Install Anaconda Distribution. Anaconda contains all the essential programs to run the functions:
   numpy, matplotlib, etc. https://www.anaconda.com/

2) Install the solver Gurobi. A free academic license of Gurobi is available. http://www.gurobi.com/

3) Connect your Gurobi version to Python. Follow the steps in this link: https://www.gurobi.com/documentation/7.5/quickstart_windows/installing_the_anaconda_py.html#section:Anaconda

Note: This version is designed for Python 3.3 and Gurobi v7.5.
