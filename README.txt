# Companion code to BML Report
# DE SALABERRY SELKAK, Gregory
# ABELES, Baptiste
#####################

# quickfigures.py is meant to be a fast way for instructors
# to reproduce all of the figures from other files at their convenience.
## NB - the figure 1 submitted in the report was computed with 1e4
## MC iterations for each case (1e5 for algorithm 0).
## By default, this file scales down these algorithms by 0.01
## Set global_scale to 100 to reproduce the figures as submitted.


# easy_samplers.py contains 4 sampling algorithms:
## 0. uniform_dist()   : determinant distribution under uniform distribution
## 1. naive_mc()       : determinant distribution under DPP via algorithm 1
## 1. pseudoGB_MC()    : determinant distribution under DPP via algorithm 2
## 2. DPPMC_dist()     : determinant distribution under DPP via DPPy package
# figure 1 

# choromanski.py
## choromanski()       : determinant distribution under DPP via algorithm 3
## chromanski_sample() : algorithm 3
## figure2_3()         : produces figure 2-3
## figure 4            : produces figure 4


