"""
Implementations of Algorithms 1,2, DPPy and Reference
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP

N, d = 10, 2
res = 200-1
# Jacobi parameters in [-0.5, 0.5]^{d x 2}
#jac_params = np.array([[0.5, 0.5],
#                       [-0.3, 0.4]])

jac_params = np.zeros((2,2))
dpp = MultivariateJacobiOPE(N, jac_params)

def visualize_kernel():
    def coordinate_square(res,d=2):
        sqrI = np.indices((np.ones(d, dtype=int)*res))/int((res+1)/2)
        sqrI-=sqrI.mean()
        return sqrI
    sqrI = coordinate_square(res, d)
    general = np.zeros(sqrI[0].shape)
    for x in range(res):
        for y in range(res):
            general[x,y] = dpp.K(sqrI[1,x,y],sqrI[0,x,y])
    plt.matshow(general[:,::-1])
#%%
def plot_determinant_dists(ref_output, _your_output, names=None):
    # plotting code for determinant distributions
    bins=np.linspace(-40,20,180)

    if len(_your_output)>10:
        your_output = np.expand_dims(_your_output,axis=0)
    else:
        your_output = _your_output
    fig, ax = plt.subplots(len(your_output)+1, 2, sharex=True)

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
#%%    

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
    detdist = np.zeros((outnb, rho))
    for Z in range(outnb):
        for z in range(rho):
            samples = np.random.uniform(-1,1,size=(ope.N,ope.dim))
            gram = np.zeros((samples.shape[0],samples.shape[0]))
            for i in range(samples.shape[0]):
                for ii in range(samples.shape[0]):
                    gram[i,ii] = ope.K(samples[i],samples[ii])
            detdist[Z,z]=np.linalg.det(gram)
        probarr = detdist[Z]/detdist[Z].sum()
        idx = rv_discrete(values=(np.arange(len(probarr)),probarr)).rvs()
        detvals[Z] = detdist[Z,idx]

    return detvals


def pseudoGB_MC(ope, outnb):
    # Algorithm 2
    rho = ope.N*3
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


def main():
    N, d = 10, 2
    res = 200-1
    ambient = uniform_dist(dpp, int(1e5))
    DPPMCoutput = DPPMC_dist(dpp,int(1e4))
    naiveMC_output = naive_MC(dpp,int(1e4))
    gb_output = pseudoGB_MC(dpp, int(1e4))
    plot_determinant_dists(ambient, [naiveMC_output, gb_output, DPPMCoutput,], \
                           ["direct MC", "pseudo-Gibbs", "dPPy.sample()"])
    plt.show()

def gen_randidx(length, maxidx=N):
    # (length) DISTINCT integers in [0, maxidx)
    indices = np.random.randint(0,maxidx,size=length)
    for i in range(len(indices)):
        while indices[i] in np.delete(indices,i):
            indices[i] = np.random.randint(0,maxidx)
    return indices

