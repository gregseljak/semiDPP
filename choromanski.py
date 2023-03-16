#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from pydpp.dpp import DPP as swkDPP
from sklearn.cluster import KMeans


def coordinate_square(res,d=2):
    sqrI = np.indices((np.ones(d, dtype=int)*res))/int((res+1)/2)
    sqrI-=sqrI.mean()
    return sqrI

def visualize_kernel(ope):

    sqrI = coordinate_square(res, d)
    general = np.zeros(sqrI[0].shape)
    for x in range(res):
        for y in range(res):
            general[x,y] = ope.K(sqrI[1,x,y],sqrI[0,x,y])
    plt.matshow(general[:,::-1])

def choromanski_sample(ope, rho, Nk):
    samples = np.random.uniform(-1,1, size=(ope.N*rho,d))
    feature_transform = ope.eval_multiD_polynomials(samples)
    outsamples = np.zeros((N,d))
    
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
    return outsamples

def choromanski(ope, _maxiter):
    maxiter=int(_maxiter)
    detout = np.zeros(maxiter)
    rho = ope.N*100
    Nk  = ope.N*20
    for z in range(maxiter):
        matout = np.zeros((ope.N,ope.N))
        outsamples = choromanski_sample(ope, rho, Nk)
        for i in range(ope.N):
            for ii in range(ope.N):
                matout[i,ii] = ope.K(outsamples[i], outsamples[ii])
        detout[z] = np.linalg.det(matout)
    return detout


#%%
def figure2_3(ope, Nk_factor=10, fixrho=False):
    res = 199
    mycmap=plt.get_cmap(name="tab20")
    DOU = 3
    if fixrho==False:
        rho_arr = np.array([10,100,500,500])
    else:
        rho_arr = np.zeros(DOU+1)+fixrho
    rho_arr = rho_arr.astype(int)
    fig, ax = plt.subplots(DOU,3)
    for dou in range(DOU):
        rho = rho_arr[dou]
        samples = np.random.uniform(-1,1, size=(ope.N*rho,d))
        feature_transform = ope.eval_multiD_polynomials(samples)
        outsamples = np.zeros((N,d))
        #number of clusters
        Nk = ope.N*Nk_factor
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
        ax[dou, 2].scatter(outsamples[:,0], outsamples[:,1])
        
        sqrI = coordinate_square(res, d).T.reshape(res**2,d)
        FsqrI = ope.eval_multiD_polynomials(sqrI)
        bigmat = (kmeans.predict(FsqrI)).reshape(res,res)
        ax[dou, 1].matshow(bigmat[::-1],extent=[-1,1,-1,1], cmap=mycmap)
        L1_center = np.zeros((Nk,d))
        for i in range(Nk):
            idxs = np.where(kmeans.labels_.astype(int)==i)[0]
            L1_center[i] = (samples[idxs]).mean(axis=0)
        ax[dou,0].scatter(samples[:,0], samples[:,1], c="black", s=2)
        ax[dou,1].scatter(L1_center[:,0], L1_center[:,1], color="black", s=5)

        ax[dou,1].get_xaxis().tick_bottom()
        ax[dou,1].get_yaxis().set_visible(False)
        ax[dou,2].get_yaxis().set_visible(False)
        ax[dou,2].get_xaxis().set_visible(False)
        ax[dou,0].get_xaxis().set_visible(False)
        ax[dou,1].get_xaxis().set_visible(False)
        pad = 5
        if fixrho==False:
            ax[dou, 2].annotate(r"$\rho=$"+str(int(rho_arr[dou])), xy=(1.3, 0.5), xytext=(pad,0),
                xycoords='axes fraction', textcoords='offset points', 
                ha='center', va='baseline')
            fig.suptitle("Coreset Sampling (Algorithm 3)\n"+r"$M=$"+str(int(Nk)), y=1.1)
        else:
            fig.suptitle("Coreset Sampling (Algorithm 3) \n"+\
                 r"$\rho=$"+str(int(rho_arr[0])), y=1.1)
        ax[dou,0].set_aspect(1)
        ax[dou,1].set_aspect(1)
        ax[dou,2].set_aspect(1)
    ax[-1,2].get_xaxis().set_ticks([-1,0,1])
    pad = 5
    ax[0,0].annotate("Oversample", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    ax[0,1].annotate("Clustering", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')
    ax[0,2].annotate("Subsample (output)", xy=(0.5, 1.), xytext=(0, pad),
            xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline')

    fig.subplots_adjust(wspace=-0.4)
    plt.show()


def figure4(ope):

    ope = dpp
    dppy_samples  = np.zeros((int(1e3*ope.N),2))
    choro_samples2 = np.zeros((int(1e3*ope.N),2))
    choro_samples3 = np.zeros((int(1e3*ope.N),2))
    for n in range(int(1e2)):
        dppy_samples[int(ope.N*n):int(ope.N*(n+1))]  = ope.sample()
        choro_samples2[int(ope.N*n):int(ope.N*(n+1))] = choromanski_sample(ope,rho= 100, Nk=20)
        choro_samples3[int(ope.N*n):int(ope.N*(n+1))] = choromanski_sample(ope,rho=1000,Nk=20)

    fig, ax = plt.subplots(1,3)
    cm = plt.cm.get_cmap('coolwarm')
    from matplotlib.colors import Normalize
    histres = 50
    cnorm = Normalize(vmin=-2, vmax=20)
    a = ax[0].hist2d(dppy_samples[:,0],dppy_samples[:,1], \
        bins=np.linspace(-1,1,histres), norm=cnorm)
    b = ax[1].hist2d(choro_samples2[:,0],choro_samples2[:,1],
        bins=np.linspace(-1,1,histres), norm=cnorm)
    c = ax[2].hist2d(choro_samples3[:,0],choro_samples3[:,1], \
        bins=np.linspace(-1,1,histres), norm=cnorm)


    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    ax[2].set_aspect(1)
    ax[0].set_title("DPPy sampler")
    ax[1].set_title("Coreset sampler\n"+r"$\rho=100$")
    ax[2].set_title("Coreset sampler\n"+r"$\rho=1000$")
    ax[1].get_yaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    fig.suptitle("Histograms on domain for DPPy and Coreset Samplers", y=0.85)
    fig.colorbar(b[3], cmap=cm, norm=cnorm, orientation='horizontal')
    plt.show()


# %%
