# Companion code to BML Report
# DE SALABERRY SELKAK, Gregory
# ABELES, Baptiste
#####################

# quickfigures.py is meant to be a fast way for instructors
# to reproduce all of the figures from other files at their convenience.
## IMPORTANT - the figures submitted in the report were computed 1e4
## MC iterations each (1e5 for algorithm 0).
## By default, this file scales down these algorithms by 0.01
## Set global_scale to 100 to reproduce the figures as submitted.


# easy_samplers.py contains 4 sampling algorithms:
## 0. uniform_dist()   : determinant distribution under uniform distribution
## 1. naive_mc()       : determinant distribution under DPP via algorithm 1
## 1. pseudoGB_MC()    : determinant distribution under DPP via algorithm 2
## 2. DPPMC_dist()     : determinant distribution under DPP via DPPy package

# choromanski.py
## choromanski()       : determinant distribution under DPP via algorithm 3
## choromanski_visualization() : produces figure 3


