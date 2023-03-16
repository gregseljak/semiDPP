#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP
from sklearn.cluster import KMeans
N, d = 10, 2
res = 200-1
jac_params = np.zeros((2,2))
dpp = MultivariateJacobiOPE(N, jac_params)

def coordinate_square(res,d=2):
    sqrI = np.indices((np.ones(d, dtype=int)*res))/int((res+1)/2)
    sqrI-=sqrI.mean()
    return sqrI

def visualize_kernel():

    sqrI = coordinate_square(res, d)
    general = np.zeros(sqrI[0].shape)
    for x in range(res):
        for y in range(res):
            general[x,y] = dpp.K(sqrI[1,x,y],sqrI[0,x,y])
    plt.matshow(general[:,::-1])

def choromanski(ope, maxiter):
    detout = np.zeros(maxiter)
    for z in range(maxiter):
        matout = np.zeros((ope.N,ope.N))
        samples = np.random.uniform(-1,1, size=(ope.N*100,d))
        feature_transform = ope.eval_multiD_polynomials(samples)
        outsamples = np.zeros((N,d))
        Nk = ope.N*20
        kmeans = KMeans(n_clusters=Nk, random_state=0, n_init="auto").fit(feature_transform)
        gram_mat = np.zeros((Nk,Nk))
        for i in range(len(gram_mat)):
            for ii in range(len(gram_mat[0])):
                gram_mat[i,ii] = kmeans.cluster_centers_[i]@kmeans.cluster_centers_[ii]
        Lens = swkDPP()
        Lens.A = gram_mat
        idxs = Lens.sample_k(ope.N)
        for i in range(ope.N):
            group = samples[kmeans.labels_==idxs[i]]
            outsamples[i] = group[np.random.randint(0,len(group)),:]
        for i in range(ope.N):
            for ii in range(ope.N):
                matout[i,ii] = ope.K(outsamples[i], outsamples[ii])
        detout[z] = np.linalg.det(matout)
    return detout
output_choro = choromanski(dpp, 1000)
#%%

plt.show()
#%%
def choromanski_visualization(ope):
    rho = 1000
    mycmap=plt.get_cmap(name="tab20")
    fig, ax = plt.subplots(1,3)
    samples = np.random.uniform(-1,1, size=(ope.N*rho,d))
    feature_transform = ope.eval_multiD_polynomials(samples)
    outsamples = np.zeros((N,d))
    #number of clusters
    Nk = ope.N*10
    kmeans = KMeans(n_clusters=Nk, random_state=0, n_init="auto").fit(feature_transform)
    gram_mat = np.zeros((Nk,Nk))
    for i in range(len(gram_mat)):
        for ii in range(len(gram_mat[0])):
            gram_mat[i,ii] = kmeans.cluster_centers_[i]@kmeans.cluster_centers_[ii]
    Lens = swkDPP()
    Lens.A = gram_mat
    idxs = Lens.sample_k(ope.N)
    for i in range(ope.N):
        group = samples[kmeans.labels_==idxs[i]]
        outsamples[i] = group[np.random.randint(0,len(group)),:]
    ax[2].scatter(outsamples[:,0], outsamples[:,1])
    
    sqrI = coordinate_square(res, d).T.reshape(res**2,d)
    FsqrI = ope.eval_multiD_polynomials(sqrI)
    bigmat = (kmeans.predict(FsqrI)).reshape(res,res)
    ax[1].matshow(bigmat[::-1],extent=[-1,1,-1,1], cmap=mycmap)
    L1_center = np.zeros((Nk,d))
    for i in range(Nk):
        idxs = np.where(kmeans.labels_.astype(int)==i)[0]
        L1_center[i] = (samples[idxs]).mean(axis=0)
    ax[0].scatter(samples[:,0], samples[:,1], c="black", s=2)
    ax[1].scatter(L1_center[:,0], L1_center[:,1], color="black", s=5)
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    ax[2].set_aspect(1)
    ax[1].get_xaxis().tick_bottom()
    ax[1].get_yaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].get_xaxis().set_ticks([-1,0,1])
    pad = 5
    ax[0].annotate("Oversample", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    ax[1].annotate("Clustering", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    ax[2].annotate("Subsample (output)", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    fig.tight_layout()
    plt.show()
choromanski_visualization(dpp)
# %%
