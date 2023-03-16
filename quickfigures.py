import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP


N, d = 10, 2
jac_params = np.zeros((d,2))
dpp = MultivariateJacobiOPE(N, jac_params)

# figure 1
import easy_samplers as ESP
ESP.figure1(scale=1)

#figure 2
import choromanski as CRM
CRM.figure2_3(dpp)
CRM.figure2_3(dpp, fixrho=100)
CRM.figure4(dpp)