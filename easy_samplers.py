#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP
#%%
N, d = 10, 2
res = 200-1
# Jacobi parameters in [-0.5, 0.5]^{d x 2}
#jac_params = np.array([[0.5, 0.5],
#                       [-0.3, 0.4]])
def coordinate_square(res,d=2):
    sqrI = np.indices((np.ones(d, dtype=int)*res))/int((res+1)/2)
    sqrI-=sqrI.mean()
    return sqrI
sqrI = coordinate_square(res, d)

#%%
general = np.zeros(sqrI[0].shape)
jac_params = np.zeros((2,2))
dpp = MultivariateJacobiOPE(N, jac_params)

#%%
for x in range(res):
    for y in range(res):
        general[x,y] = dpp.K(sqrI[1,x,y],sqrI[0,x,y])
plt.matshow(general[:,::-1])
#%%
def plot_determinant_dists(ref_output, _your_output):
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
    print(f"{len(yourhist[0])}")
    print(len(refvals))
    for i in range(len(your_output)):
        values = yourhist[i]/refvals
        ax[i,1].xaxis.set_ticklabels([])
        ax[i,1].yaxis.set_ticklabels([])
        ax[i,1].bar(bins[:-1], values)
        expect_det = bins[np.exp(bins)<1.5*values.max()]
        ax[i,1].plot(expect_det, np.exp(expect_det), color="gray", linestyle="-", label="scaled linear")
        ax[i,1].plot(0,)
        if i == 0:
            pad = 5
            ax[0,0].annotate("Distribution (log det)", xy=(0.5, 1), xytext=(0, pad),
                             xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')
            ax[0,1].annotate("Ratio against reference", xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
            ax[0,1].legend()
    ax[-1,1].bar(bins[:-1],0.4,color="green")
    ax[-1,1].yaxis.set_ticklabels([])
    ax[-1,1].set_ylim(0,1)
    ax[-1, 0].xaxis.set_ticklabels(np.arange(bins.min(), bins.max(),  int((bins.max()-bins.min())/5)))
    ax[-1,0].annotate("Distribution (log det) on reference", xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    ax[-1,1].annotate("(Reference)", xy=(0.5, 1), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    fig.suptitle("Monte Carlo - Determinant Dist ")
    fig.tight_layout()
    plt.show()
    

def uniform_dist(ope, maxiter=int(1e5)):
    output = np.zeros(int(maxiter))
    for o in range(len(output)):
        samples = np.random.uniform(-1,1, size=(ope.N,ope.dim))
        mcmat = np.zeros((ope.N, ope.N))
        for i in range(len(mcmat)):
            for ii in range(len(mcmat[i])):
                mcmat[i,ii] = ope.K(samples[i], samples[ii])
        output[o]=np.linalg.det(mcmat)
    return output
#output_flat = uniform_dist(dpp)


def DPPMC_dist(ope, maxiter=int(1e3)):
    # Monte Carlo for determinant distribution
    output = np.zeros(int(maxiter))
    for o in range(len(output)):
        samples = ope.sample()
        mcmat = np.zeros((samples.shape[0], samples.shape[0]))
        for i in range(len(mcmat)):
            for ii in range(len(mcmat)):
                mcmat[i,ii] = ope.K(samples[i], samples[ii])
            output[o] = np.linalg.det(mcmat)
    return output
#output - 1e5
DPPMCoutput = DPPMC_dist(dpp)
#%%
def naive_MC(ope, outnb):
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
naiveMC_output = naive_MC(dpp,1000)
#%%
def pseudoGB_MC(ope, outnb):
    rho = ope.N*3
    detvals = np.zeros(outnb)
    kDPP = swkDPP()
    for Z in range(outnb):
        supersample = np.random.uniform(-1,1,size=(rho, ope.dim))
        kDPP.A = np.zeros((rho,rho))
        for i in range(rho):
            for ii in range(rho):
                kDPP.A[i,ii] = ope.K(supersample[i], supersample[ii])
        if np.sum(kDPP.A ==0)>0:
            print("Greg you dunce")
            print(np.where(kDPP.A==0))
            return None
        idx = kDPP.sample_k(ope.N)
        detvals[Z] = np.linalg.det(kDPP.A[idx][:,idx])
    return detvals
gb_output = pseudoGB_MC(dpp, int(1e4))
#%%
plt.hist(np.log(gb_output),bins=np.linspace(-40,20,180))
#%%
plot_determinant_dists(output_flat, [naiveMC_output, DPPMCoutput,gb_output])
#%%

#%%
def gen_randidx(length, maxidx=N):
    # (length) DISTINCT integers in [0, maxidx)
    indices = np.random.randint(0,maxidx,size=length)
    for i in range(len(indices)):
        while indices[i] in np.delete(indices,i):
            indices[i] = np.random.randint(0,maxidx)
    return indices
def minor_determinants(bigmat, minorsize):
    matN = bigmat.shape[0]
    mc_iter = matN**2
    dets = np.zeros(mc_iter)
    for i in range(mc_iter):
        elim = matN-minorsize
        exclude = gen_randidx(elim, matN)
            
        include = np.delete(np.arange(0,matN), exclude)
        #vals, vecs = np.linalg.eig(bigmat[include][:,include])
        
        dets[i] = np.linalg.det(bigmat[include][:,include])
    plt.hist(dets/minorsize)


# %%
plt.hist(detvals)
# %%
