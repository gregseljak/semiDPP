"""
Implementations of Algorithms 0,1,2, and DPPy
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP
import choromanski as CRM


def plot_determinant_dists(ref_output, _your_output, names=None):
    # plotting code for determinant distributions
    bins=np.linspace(-40,20,180)
    lsxlim = [-40,15]
    rsxlim = [-10,15]
    if len(_your_output)>10:
        your_output = np.expand_dims(_your_output,axis=0)
    else:
        your_output = _your_output
    fig, ax = plt.subplots(len(your_output)+1, 2, sharex=False, figsize=(10, 7.5))

    yourhist = np.empty(len(your_output),dtype=object)
    for i in range(len(your_output)):
        yourhist[i] = ax[i,0].hist(np.log(your_output[i]), bins=bins, density=True)
        yourhist[i] = yourhist[i][0]
    refhist = ax[-1,0].hist(np.log(ref_output),bins=bins, density=True, color="green")
    refvals = refhist[0]
    refvals[refvals==0] = refvals[refvals>0].min()

    for i in range(len(your_output)):
        pad = 5
        values = yourhist[i]/refvals
        ax[i,1].xaxis.set_ticklabels([])
        ax[i,1].yaxis.set_ticklabels([])
        ax[i,1].bar(bins[:-1], values)
        expect_det = bins[np.exp(bins)<1.5*values.max()]
        ax[i,1].plot(expect_det, np.exp(expect_det), color="gray", linestyle="-", label="scaled linear")
        ax[i,1].set_xlim(rsxlim)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].set_xlim(lsxlim)
        if names is not None:
            ax[i,1].annotate(names[i], xy=(1.2, 0.5), xytext=(pad,0),
                                xycoords='axes fraction', textcoords='offset points', 
                                ha='center', va='baseline')
        if i == 0:
            
            ax[0,0].annotate("Distribution (log det)", xy=(0.5, 1), xytext=(0, pad),
                             xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')
            ax[0,1].annotate("Ratio against $\pi_\mu$", xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
            ax[0,1].legend()
    ax[-1,1].bar(bins[:-1],0.4,color="green")
    ax[-1,1].yaxis.set_ticklabels([])
    ax[-1,1].set_ylim(0,1)
    ax[-1,0].set_xlim(lsxlim)
    ax[-1,1].set_xlim(rsxlim)
    ax[-1,1].annotate(r"Ambient $\pi_\mu$", xy=(1.2, 0.5), xytext=(pad,0),
                                xycoords='axes fraction', textcoords='offset points', 
                                ha='center', va='baseline')
    ax[-1, 0].xaxis.set_ticklabels(np.arange(bins.min(), bins.max(),  int((bins.max()-bins.min())/5)))
    ax[-1,0].annotate("", xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    ax[-1,1].annotate("", xy=(0.5, 1), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    fig.suptitle("Monte Carlo - Determinant Dist ")
    fig.text(0.00, 0.5, 'frequency', va='center', rotation='vertical')
    fig.text(0.5, 0.00, 'log det', ha='center')
    fig.tight_layout()

    plt.show()


def uniform_dist(ope, maxiter=int(1e5)):
    ### MC To get distribution of determinants under ambient
    output = np.zeros(int(maxiter))
    for o in range(len(output)):
        samples = np.random.uniform(-1,1, size=(ope.N,ope.dim))
        mcmat = np.zeros((ope.N, ope.N))
        for i in range(len(mcmat)):
            for ii in range(len(mcmat[i])):
                mcmat[i,ii] = ope.K(samples[i], samples[ii])
        output[o]=np.linalg.det(mcmat)
    return output


def DPPMC_dist(ope, maxiter=int(1e3)):
    # DPPy sampling method
    output = np.zeros(int(maxiter))
    for o in range(len(output)):
        samples = ope.sample()
        mcmat = np.zeros((samples.shape[0], samples.shape[0]))
        for i in range(len(mcmat)):
            for ii in range(len(mcmat)):
                mcmat[i,ii] = ope.K(samples[i], samples[ii])
            output[o] = np.linalg.det(mcmat)
    return output

def naive_MC(ope, outnb):
    # Algorithm 1
    rho = ope.N*20
    detvals = np.zeros(outnb)
    detdist = np.zeros(rho)
    for Z in range(outnb):
        for z in range(rho):
            samples = np.random.uniform(-1,1,size=(ope.N,ope.dim))
            gram = np.zeros((samples.shape[0],samples.shape[0]))
            for i in range(samples.shape[0]):
                for ii in range(samples.shape[0]):
                    gram[i,ii] = ope.K(samples[i],samples[ii])
            detdist[z]=np.abs(np.linalg.det(gram))
        probarr = np.abs(detdist/detdist.sum())
        idx = rv_discrete(values=(np.arange(len(probarr)),probarr)).rvs()
        detvals[Z] = detdist[idx]

    return detvals


def pseudoGB_MC(ope, outnb):
    # Algorithm 2
    rho = ope.N*10
    detvals = np.zeros(outnb)
    kDPP = swkDPP()
    for Z in range(outnb):
        supersample = np.random.uniform(-1,1,size=(rho, ope.dim))
        kDPP.A = np.zeros((rho,rho))
        for i in range(rho):
            for ii in range(rho):
                kDPP.A[i,ii] = ope.K(supersample[i], supersample[ii])
        idx = kDPP.sample_k(ope.N)
        detvals[Z] = np.linalg.det(kDPP.A[idx][:,idx])
    return detvals

def figure1():
    N, d = 10, 2
    res = 200-1
    jac_params = np.zeros((2,2))
    dpp = MultivariateJacobiOPE(N, jac_params)
    naiveMC_output = naive_MC(dpp,int(1e4))
    ambient = uniform_dist(dpp, int(1e5))
    DPPMCoutput = DPPMC_dist(dpp,int(1e4))

    gb_output = pseudoGB_MC(dpp, int(1e4))
    choromanski_output = CRM.choromanski(int(1e4))
    plot_determinant_dists(ambient, [naiveMC_output, gb_output, DPPMCoutput,choromanski_output], \
                        ["direct MC", "pseudo-Gibbs", "dPPy.sample()", "choromanski"])
    plt.show()

def figure_1(scale=1):
    N, d = 10, 2
    jac_params = np.zeros((d,2))
    dpp = MultivariateJacobiOPE(N, jac_params)
    naiveMC_output = naive_MC(dpp,int(scale*1e2))
    ambient = uniform_dist(dpp, int(scale*1e3))
    DPPMCoutput = DPPMC_dist(dpp,int(scale*1e2))
    gb_output = pseudoGB_MC(dpp, int(scale*1e2))
    choromanski_output = CRM.choromanski(dpp,int(scale*1e2))

    plot_determinant_dists(ambient, [naiveMC_output, gb_output, DPPMCoutput,choromanski_output], \
                            ["direct MC", "pseudo-Gibbs", "dPPy.sample()", "choromanski"])
    plt.show()

def gen_randidx(length, maxidx=N):
    # (length) DISTINCT integers in [0, maxidx)
    indices = np.random.randint(0,maxidx,size=length)
    for i in range(len(indices)):
        while indices[i] in np.delete(indices,i):
            indices[i] = np.random.randint(0,maxidx)
    return indices

