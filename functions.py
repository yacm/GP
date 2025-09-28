import numpy as np 
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import scipy.integrate as integrate
from torch.special import gammaln
from torch.autograd.functional import hessian
import scipy.integrate as integrate
import h5py as h5
from scipy.optimize import minimize 
from scipy import special 
from scipy.optimize import Bounds 
from scipy.linalg import cho_solve 
import time
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline 
import scipy.special
import datetime
from GP import *
import re

def get_dist_matelem(z, p, t_min,ITD="Re"):
    f = 0
    if p <= 3:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.unphased.hdf5','r')
    else:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.phased-d001_2.00.hdf5','r')
    M_z_p = np.array(f['MatElem/bins/'+ITD+'/mom_0_0_+'+str(p)+'/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_z_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_p = np.array(f['MatElem/bins/Re/mom_0_0_+'+str(p)+'/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    
    f.close()
    return M_z_p * M_0_0 / M_0_p / M_z_0

def get_final_res(z, p,ITD):
    m_4, _ = get_dist_matelem(z, p, 4,ITD)
    m_6, s_6 = get_dist_matelem(z, p, 6,ITD)
    m_8, s_8 = get_dist_matelem(z, p, 8,ITD)
    return m_6, np.sqrt(s_6**2)#+(m_4-m_6)**2)

def get_data(ITD):
    Np = 6
    Nz = 12
    Nj = 349
    rMj = np.empty([Nj,Np,Nz])
    nu = np.empty([Np,Nz])
    for p in range(1,Np+1):
        for z in range (1,Nz+1):
            nu[p-1,z-1] = 2.0*np.pi/32.0 *p *z
            #print(p,z,nu[p-1,z-1])
            m_4 = get_dist_matelem(z,p,4,ITD)
            m_6 = get_dist_matelem(z,p,6,ITD)
            m_8 = get_dist_matelem(z,p,8,ITD)
            #expo fit
            m = (m_4*m_8 - m_6**2)/(m_4 + m_8 - 2 * m_6)
            # this fails for certain cases where the denomenator goes too close to zero
            # use the m_6 as default
            rMj[:,p-1,z-1] = m_6
            #Nj=m.shape[0]
            #print(z,p,np.mean(m_4),np.mean(m_6),np.mean(m_8), np.mean(m),np.std(m)*np.sqrt(Nj-1))
    rM = np.mean(rMj,axis=0)
    rMe = np.std(rMj,axis=0)*np.sqrt(Nj) 
    return nu,rMj,rMe,rM

def restart(fits_comb):
    number=len(fits_comb)
    for i in range(0,number):
        fits_comb[i].hyperparametersvalues()
def train_model(fits_comb,Ntrain,function,lr,mode):
    start=time.time()
    number=len(fits_comb)
    for i in range(0,number):
        #mode =fits_comb[i].mode
        #print(fits_comb[i].mode)
        if i in [12,13,14]:
            fits_comb[i].train(Ntrain,lr=lr,mode=mode,function=function)
        else:
            fits_comb[i].train(1,lr=1e-4,mode=mode,function=function)
        print(tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args  + (fits_comb[i].sig,)),flush=True)
    end=time.time()
    print("time",end-start)


def plottrained(fit,iB,nn,zs):

    fCI=1.00
    number=15
    PDFpos=1
    ITDpos=0
    col = ["#332288","#117733","#EF8738", "#CC6677","#88CCEE", "#882255","#44AA99","#999933","#AA4499", "#661100", "#6699CC", "#F0E442","#332288","#117733","#EF8738", "#CC6677","#88CCEE", "#882255","#44AA99","#999933","#AA4499", "#661100", "#6699CC", "#F0E442",]
    col
    for z in zs:
        fig,ax=plt.subplots(1,2,figsize=(18, 10.5))
        x_grid=fit[z].x_grid

        p,Cp = fit[z].ComputePosterior()
        p,Cp = p.to("cpu"),Cp.to("cpu")
        Cp = 0.5*(Cp+Cp.T)
        svd=np.linalg.svd(Cp)
        Cp=svd[0] @ np.diag(np.abs(svd[1])) @ svd[2]
        Cp=0.5*(Cp+Cp.T) +1e-6*np.eye(Cp.shape[0])#regulate the posterior covariance
        pdfMc= np.random.multivariate_normal(p,Cp,(500,))
        ax[0].plot(x_grid,p,label=fit[z].name+' training steps: '+str(fit[z].trainingcount),color=col[z])
        perror=1.0*np.diag(Cp)**(0.5)
        #p=fits_comb[i].Pd(fits_comb[i].x_grid.cpu(),*fits_comb[i].pd_args).numpy()
        ax[0].fill_between(x_grid, p - perror, p + perror, facecolor=col[z], alpha=0.3)
        ax[0].legend()
        ax[0].set_ylim([-0.3,6])
        ttQ = pdfMc@iB.T
        covnu= np.cov(ttQ.T)
        mttQ = ttQ.mean(axis=0)
        #ax[1].plot(nn,ttQ.T,color=col[z],alpha=0.05)
        ax[1].plot(nn,mttQ,color=col[z])
        ax[1].fill_between(nn, mttQ - fCI*np.sqrt(np.diag(covnu)), mttQ + fCI*np.sqrt(np.diag(covnu)), facecolor=col[z], alpha=0.3)
        if z >=12:
            if fit[z].ITD=="Re":
                MMM='real'
            elif fit[z].ITD=="Im":
                MMM='imag'
            if z==12:
                datanu=np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_itd_'+MMM+'_numax4.dat',dtype=np.float64)
            elif z==13:
                datanu=np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_itd_'+MMM+'_numax10.dat',dtype=np.float64)
            elif z==14:
                datanu=np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_itd_'+MMM+'_numax25.dat',dtype=np.float64)

            nu_d_grid = datanu.T[1]
            
            M=datanu.T[2:].mean(axis=0)
            eMnu=datanu.T[2:].std(axis=0)#*np.sqrt(M.shape[0]-1)
            ax[1].errorbar(nu_d_grid,M,eMnu,fmt='.',alpha=0.5,label=fit[z].name,color='red')
            #save this plots fig
            ax[1].set_xlabel(r"$n_\nu$")
            ax[1].set_ylabel(r"$\langle I_{n_\nu} \rangle$")
            ax[0].set_xlabel(r"$x$")
            ax[0].set_ylabel(r"$f(x)$")
            ax[0].set_title("PDF")
            ax[1].set_title("ITD")
            ax[1].legend()
            ax[0].legend()
    #fig.savefig(fit[z].name+'_grid'+fit[z].gridname+'.pdf',bbox_inches='tight')    

def Modelaveraging_importantsampling_gauss(fits_comb,nn,Nsamp,iB,list):
    samplesqx=[]
    samplesQv=[]
    qofxs=[]
    Qofvs=[]
    covs=[]
    covsnu=[]
    print(f'### INITIALIZE MODEL AVERAGING {fits_comb[0].modelname}_{fits_comb[0].kernelname}({fits_comb[0].ITD}_{fits_comb[0].mode}) ###')
    #remember that for Im(M) the range goes to 12 because there is no mock data set
    for i in range(len(fits_comb)):
        if fits_comb[i].name in ['z=NNPDF(4)','z=NNPDF(10)','z=NNPDF(25)']:
            #fits_comb[i].trace=filtered_traces[i][::filtered_traces[i].shape[0]//nnn]
            qofx,cov,pms,Qofv,covnu,Qvs=fits_comb[i].model_averaging_IS(Nsamp,nn,iB,prob=True,fullevi=True)
        else:
            #placeholder for the other models
            #fits_comb[i].trace=filtered_traces[i][::filtered_traces[i].shape[0]//10]
            qofx,cov,pms,Qofv,covnu,Qvs=fits_comb[i].model_averaging_IS(2,nn,iB,prob=True,fullevi=True)
        pms=tr.stack(pms)
        Qvs=tr.stack(Qvs)
        #samplesqx.append(pms)
        #samplesQv.append(Qvs)
        qofxs.append(qofx)
        Qofvs.append(Qofv)
        covs.append(cov)
        covsnu.append(covnu)
        print("Model Averaging Important Sampling (2nd level)",fits_comb[i].name,"done",flush=True)
    #return samplesqx,samplesQv,qofxs,Qofvs,covs,covsnu
    return qofxs, Qofvs, covs, covsnu


def save_data(fits_comb,data):
    qofxs, Qofvs, covs, covsnu = data

    for i in range(len(fits_comb)):
        #fits_comb[i].trace=filtered_traces[i][::filtered_traces[i].shape[0]//nnn]
        filepath = os.path.join(
            f"{fits_comb[i].modelname}_{fits_comb[i].kernelname}({fits_comb[i].mode}+{fits_comb[i].gridname})",
            f"data_{fits_comb[i].ITD}"
        )
        #create directory if it does not exist
        try:
            #permisions for creating directory
            os.makedirs(filepath, exist_ok=True, mode=0o755)
        except Exception as e:
            print(f"Error creating directory {filepath}: {e}")
            continue
        try:
            #save samplesqx[i].to("cpu")
            #tr.save(samplesqx[i].to("cpu"), f"{filepath}/samplesqx_{i+1}.pt")
            #tr.save(samplesQv[i].to("cpu"), f"{filepath}/samplesQv_{i+1}.pt")
            tr.save(qofxs[i].to("cpu"), f"{filepath}/qofx_{fits_comb[i].name}.pt")
            tr.save(Qofvs[i].to("cpu"), f"{filepath}/Qofv_{fits_comb[i].name}.pt")
            tr.save(covs[i].to("cpu"), f"{filepath}/cov_{fits_comb[i].name}.pt")
            tr.save(covsnu[i].to("cpu"), f"{filepath}/covnu_{fits_comb[i].name}.pt")
        except Exception as e:
            print(f"Error saving data for {fits_comb[i].ITD} {fits_comb[i].name}, {fits_comb[i].modelname}, {fits_comb[i].kernelname} {fits_comb[i].mode} {fits_comb[i].gridname}: {e}")
            continue


def redefine_kernel(fit_test,iB,pval):
    numax=-1
    for i in range(len(fit_test)):
        if fit_test[i].name in ['z=NNPDF(4)','z=NNPDF(10)','z=NNPDF(25)']:
            flag=False
            print("##### Name of the fit",fit_test[i].name,flush=True)
            #transform to tuple
            DeltaM=tr.diag(fit_test[i].Gamma).numpy()[numax]**0.5
            absM=np.abs(fit_test[i].Y.numpy()[numax])
            maxi=np.maximum((1+pval)*DeltaM,pval*absM)
            sigeps=tr.linspace(1.0,5.0,50)
            leps=tr.linspace(1.0,0.05,20)
            for jj in range(0,sigeps.shape[0]):
                for kk in range(0,leps.shape[0]):
                    #print("type",DeltaM.dtype,absM.dtype)
                    #shapes
                    #print("shapes",DeltaM.shape,absM.shape,DeltaM,absM)

                    #print("maxi",maxi)
                    p,Cp= fit_test[i].ComputePosterior()
                    p,Cp = p.to("cpu"),Cp.to("cpu")
                    #print("p",p.shape,"Cp",Cp.shape)
                    Cp = 0.5*(Cp+Cp.T)
                    svd=np.linalg.svd(Cp)
                    Cp=svd[0] @ np.diag(np.abs(svd[1])) @ svd[2]
                    Cp=0.5*(Cp+Cp.T) +1e-6*np.eye(Cp.shape[0])#regulate the posterior covariance
                    pdfMc= np.random.multivariate_normal(p,Cp,(500,))
                    M=pdfMc@ iB.T
                    ttQ = pdfMc@iB.T
                    covnu= np.cov(ttQ.T)
                    mttQ = ttQ.mean(axis=0)


                    DeltaM_th= np.max(np.diag(covnu)**0.5).item()
                    #print("deltaM_th",DeltaM_th,"DeltaM",DeltaM,"absM",absM)
                    if DeltaM_th>=maxi:
                        flag=True
                        break
                    
                    else:
                        fit_test[i].ker_args=tr.tensor(fit_test[i].ker_args)
                        fit_test[i].ker_args[0]=sigeps[jj]**2
                        fit_test[i].ker_args[1]=leps[kk]
                        continue
                if flag:
                    print("values of the params",fit_test[i].ker_args,flush=True)
                    print("pass the test",flush=True)
                    fit_test[i].ker_args=tuple(fit_test[i].ker_args)
                    print("Parameters changed")
                    break
                #fit_test[i].ker_args[0]=tr.tensor(3.0)
                #fit_test[i].ker_args[1]=tr.log(tr.log(tr.tensor(2.5)))
                #print(fit_test[i].ker_args)
                #print(DeltaM_th,(1+p)*DeltaM,p*absM)

            #print(fit_test[i].ker_args[0],fit_test[i].ker_args[1])
            fit_test[i].ker_args=tuple(fit_test[i].ker_args)
            print("fail for the model", fit_test[i].modelname,flush=True)
            print(fit_test[i].ker_args,flush=True)

    return fit_test


def load_data_usesamples(fits_comb):
    samplesqx = []
    samplesQv = []
    qofxs = []
    Qofvs = []
    covs = []
    covsnu = []

    for i in range(0, 15):
        filepath = f"/{fits_comb[i].modelname}_{fits_comb[i].kernelname}({fits_comb[i].mode}+{fits_comb[i].gridname})/data_{fits_comb[i].ITD}"
        filepath = os.path.join(
            f"{fits_comb[i].modelname}_{fits_comb[i].kernelname}({fits_comb[i].mode}+{fits_comb[i].gridname})",
            f"data_{fits_comb[i].ITD}"
        )
        try:
            samplesqx.append(tr.load(f"{filepath}/samplesqx_{i+1}.pt"))
            samplesQv.append(tr.load(f"{filepath}/samplesQv_{i+1}.pt"))
            qofxs.append(tr.load(f"{filepath}/qofx_{i+1}.pt"))
            Qofvs.append(tr.load(f"{filepath}/Qofv_{i+1}.pt"))
            covs.append(tr.load(f"{filepath}/cov_{i+1}.pt"))
            covsnu.append(tr.load(f"{filepath}/covnu_{i+1}.pt"))
        except Exception as e:
            print(f"Error loading data for {fits_comb[i].ITD} {fits_comb[i].name}, {fits_comb[i].modelname}, {fits_comb[i].kernelname} {fits_comb[i].mode} {fits_comb[i].gridname}: {e}")
            continue
    return samplesqx, samplesQv, qofxs, Qofvs, covs, covsnu

def MC(func,point,Nsamp,epsilon=None,bar=False):
    d=point.shape[0]
    trace = tr.zeros(Nsamp,d) #trace of the samples
    old_prob = func(point) #probability of the old point
    old_x = point #old point
    #delta = np.random.normal(0,0.5,Nsamp) #trial distribution
    if epsilon is None:
        #delta = tr.normal(0,0.5,(Nsamp,d))
        delta = tr.distributions.multivariate_normal.MultivariateNormal(tr.zeros(d), 0.5*tr.eye(d)).sample((Nsamp,))
    else:
        #delta = tr.normal(0,0.5,(Nsamp,d)) #trial distribution
        cov=tr.diag(epsilon)
        delta = tr.distributions.multivariate_normal.MultivariateNormal(tr.zeros(d), cov).sample((Nsamp,)) #trial distribution
        #delta = np.random.uniform(-0.5,0.5,Nsamp) #trial distribution
    
    accepted=0
    if bar==True:
        for i in tqdm(range(Nsamp)):
            new_x = old_x + delta[i]
            new_prob = func(new_x)
            acceptance = new_prob/old_prob
            if(acceptance>np.random.uniform(0,1)):
                trace[i] = new_x
                old_x = new_x
                old_prob = new_prob
                accepted=accepted+1
            else:
                trace[i] = old_x # remain in the same state
    else:
        for i in range(Nsamp):
            new_x = old_x + delta[i]
            new_prob = func(new_x)
            acceptance = new_prob/old_prob
            if(acceptance>np.random.uniform(0,1)):
                trace[i] = new_x
                old_x = new_x
                old_prob = new_prob
                accepted=accepted+1
            else:
                trace[i] = old_x
    print("Acceptance rate: ", accepted/Nsamp)
    return trace

#load trained parameters
def load_trained(fits_comb,ITD,modelname,kernelname,grid):
    number=len(fits_comb)
    basepath=f"/sciclone/scr-lst/yacahuanamedra/GP/{modelname}_{kernelname}({fits_comb[0].mode}+{grid})/min"
    for i in range(0,number):
        file_name = f"K{ITD}({fits_comb[i].name})(train).pt"
        file_path = os.path.join(basepath, file_name)
        try:
            fits_comb[i].pd_args = tr.load(file_path)[0:fits_comb[i].Npd_args]
            fits_comb[i].ker_args = tr.load(file_path)[fits_comb[i].Npd_args:]
            print(f"Trained parameters loaded from: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

def plothist(trace,mygp1,disc,params="model+kernel+noise",prior=False,burn=100,kernel='jacobi'):
    fig, ax = plt.subplots(trace.shape[1], 1, figsize=(10, 10), sharex=False, sharey=False)

    mygp=mygp1

    i0 = 100
    iF=10000
    if kernel=='jacobi':
        lab=['α', 'β', 'N','s', 't', 'a', 'b','σerror']
        labprior=['α-prior', 'β-prior', 's-prior', 't-prior', 'a-prior', 'b-prior','σerror-prior']
        if params=="kernel":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        elif params=="kernel+noise":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        else:
            pass

    elif kernel=='combinedRBF':
        lab=['α','β','N','σ1','w1','σ2','w2','s','σnoise']
        labprior=['α-prior','β-prior','σ1-prior','w1-prior','σ2-prior','w2-prior','s-prior','σerror-prior']
        if params=="kernel":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        elif params=="kernel+noise":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        else:
            pass
        
    elif kernel=='RBF':
        lab=['α','β','N','σ','w','σnoise']
        labprior=['α-prior','β-prior','σ-prior','w-prior','σerror-prior']
        if params=="kernel":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        elif params=="kernel+noise":
            lab=lab[mygp.Npd_args:]
            mygp.prior_dist=mygp.prior_dist[mygp.Npd_args:]
        else:
            pass
    elif kernel=='model':
        lab=['α','β']
        labprior=['α-prior','β-prior']
    else:
        #parameters in greek
        lab=['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
        labprior=['α-prior','β-prior','γ-prior','δ-prior','ε-prior','ζ-prior','η-prior','θ-prior','ι-prior','κ-prior','λ-prior','μ-prior','ν-prior','ξ-prior','ο-prior','π-prior','ρ-prior','σ-prior','τ-prior','υ-prior','φ-prior','χ-prior','ψ-prior','ω-prior']

    col=['red','blue','green','pink','black','orange','purple','brown','yellow','cyan','magenta','grey',"lightblue","lightgreen","lightcoral","lightpink","lightyellow","lightcyan","lightmagenta","lightgrey","darkblue","darkgreen","darkcoral","darkpink","darkyellow","darkcyan","darkmagenta","darkgrey"]
    for i in range(trace.shape[1]):
        ax[i].hist(trace[i0:iF,i],bins=disc,label=lab[i],color=col[i],density=True)
        if prior:
            initial=mygp.prior_dist[i].shift
            final=mygp.prior_dist[i].shift+mygp.prior_dist[i].scale
            xxx = tr.linspace(initial,final,1000)
            distexp=mygp.prior_dist[i]
            pdfs=tr.zeros(xxx.shape[0])
            for k in range(xxx.shape[0]):
                pdfs[k]=distexp.pdf(xxx[k])
            ax[i].plot(xxx,pdfs.detach().numpy())
            ax[i].set_xlim([initial-0.5,final+0.5])
        ax[i].legend()
    plt.show()

def plothist1(trace,mygp1,disc,params="model+kernel+noise",prior=False,burn=100,kernel='jacobi'):
    fig, ax = plt.subplots(trace.shape[1], 1, figsize=(10, 10), sharex=False, sharey=False)

    mygp=mygp1

    i0 = 100
    iF=10000
        #parameters in greek
    """    lab=['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
        labprior=['α-prior','β-prior','γ-prior','δ-prior','ε-prior','ζ-prior','η-prior','θ-prior','ι-prior','κ-prior','λ-prior','μ-prior','ν-prior','ξ-prior','ο-prior','π-prior','ρ-prior','σ-prior','τ-prior','υ-prior','φ-prior','χ-prior','ψ-prior','ω-prior']"""

    col=['red','blue','green','pink','black','orange','purple','brown','yellow','cyan','magenta','grey',"lightblue","lightgreen","lightcoral","lightpink","lightyellow","lightcyan","lightmagenta","lightgrey","darkblue","darkgreen","darkcoral","darkpink","darkyellow","darkcyan","darkmagenta","darkgrey"]
    for i in range(trace.shape[1]):
        ax[i].hist(trace[i0:iF,i],bins=disc,label=lab[i],color=col[i],density=True)
        if prior:
            initial=mygp.prior_dist[i].shift
            final=mygp.prior_dist[i].shift+mygp.prior_dist[i].scale
            xxx = tr.linspace(initial,final,1000)
            distexp=mygp.prior_dist[i]
            pdfs=tr.zeros(xxx.shape[0])
            for k in range(xxx.shape[0]):
                pdfs[k]=distexp.pdf(xxx[k])
            ax[i].plot(xxx,pdfs.detach().numpy())
            ax[i].set_xlim([initial-0.5,final+0.5])
        ax[i].legend()
    plt.show()
def plotrace(trace,burn=100,kernel='jacobi'):
    fig, ax = plt.subplots(trace.shape[1],figsize=(20, 8))
    i0 = burn
    iF=trace.shape[0]
    if kernel=='jacobifull':
        lab=['α','β','N','s','t','a','b']
    elif kernel=='rbf':
        lab=['α','β','N','σ','w','σnoise']
    else:
        lab=['α', 'β','N','σ1','w1','σ2','w2','s','σnoise']
    col=['red','blue','green','pink','black','orange','purple','brown','grey',"lightblue","lightgreen","lightcoral","lightpink","lightyellow","lightcyan","lightmagenta","lightgrey","darkblue","darkgreen","darkcoral","darkpink","darkyellow","darkcyan","darkmagenta","darkgrey"]
    for i in range(trace.shape[1]):
        ax[i].plot(trace[i0:iF,i],label=lab[i],color=col[i])
        ax[i].legend()
    plt.show()
#from tensor to list
def tensor2list(tensor):
    return [tensor[i].item() for i in range(tensor.shape[0])]


def compute_mode_per_column(samples: tr.Tensor, decimals: int = 3):
    # Round values to reduce float uniqueness
    rounded = tr.round(samples * 10**decimals) / 10**decimals
    modes = []
    counts = []

    for col in rounded.T:  # iterate over parameters
        values, freq = tr.unique(col, return_counts=True)
        max_idx = tr.argmax(freq)
        modes.append(values[max_idx])
        counts.append(freq[max_idx])

    return tr.stack(modes), tr.stack(counts)

##integrator
class FE_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,2.0*x[self.N-1] - x[self.N-2])
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
 ##       if(i==0):
 ##           R=(x- self.x[2])/(self.x[1] -self.x[2])*np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[2]-x,0.5)

            #R= self.pulse(x,self.x[0],self.x[1])
            #R= (x- self.x[0])/(self.x[1] -self.x[0])*self.pulse(x,self.x[0],self.x[1])
            #R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])
            #R+=(x- self.x[1])/(self.x[0] -self.x[1])*self.pulse(x,self.x[0],self.x[1]) 
##            return R
        ii=i+1
        R = (x- self.x[ii-1])/(self.x[ii] -self.x[ii-1])*self.pulse(x,self.x[ii-1],self.x[ii  ])
        R+= (x- self.x[ii+1])/(self.x[ii] -self.x[ii+1])*self.pulse(x,self.x[ii  ],self.x[ii+1])

       # if(i==0):
       #     R *=2
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
   
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        #res[0,:] *=2
        #res[:,0] *=2
        return res
        
    def ComputeI(self,i,Kernel):
        I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[i], self.x[i+2], epsrel=1e-12)
        self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), self.x[j], self.x[j+2],self.x[i], self.x[i+2], epsrel=1e-12)
        self.eI += eI
        return I
    
    
# quadratic finite elements are more complicated...
# ... but now it works!
# also I should try the qubic ones too. It works only up to 1e-6
class FE2_Integrator1:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,[2.0*x[self.N-1] - x[self.N-2], 3.0*x[self.N-1]-2*x[self.N-2],0] )
        #self.x = np.append([-x[0],0],xx)
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
        R=0.0
        if(i==0):

            R+=(x- self.x[2])*(x- self.x[3])/((self.x[1] -self.x[3])*(self.x[1] -self.x[2]))*np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[3]-x,0.5)
            #self.pulse(x,self.x[0],self.x[3])
            return R
        ii =i+1
        if(ii%2==0):
            R  += (x- self.x[ii-1])*(x- self.x[ii+1])/((self.x[ii] -self.x[ii+1])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-1],self.x[ii+1])
            return R
        else:
            R += (x- self.x[ii-2])*(x- self.x[ii-1])/((self.x[ii] -self.x[ii-2])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-2],self.x[ii  ])
            R += (x- self.x[ii+1])*(x- self.x[ii+2])/((self.x[ii] -self.x[ii+2])*(self.x[ii] -self.x[ii+1]))*self.pulse(x,self.x[ii  ],self.x[ii+2])
            return R
    
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
        
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency 
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        return res
    
    def ComputeI(self,i,Kernel):
        #if(i==0):
        #    I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,0), self.x[0], self.x[3])
        #    self.eI += eI
        #    return I
        ii=i+1
        if(ii%2==0):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-1], self.x[ii+1])
            self.eI += eI
        else:
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-2], self.x[ii+2])
            self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        # I need to fix the i=0 case
        ii=i+1
        jj=j+1
        if(ii%2==0):
            xx = (self.x[ii-1], self.x[ii+1])
        else:
            xx = (self.x[ii-2], self.x[ii+2])
        if(jj%2==0):
            yy = (self.x[jj-1], self.x[jj+1])
        else:
            yy = (self.x[jj-2], self.x[jj+2])
        
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), yy[0], yy[1],xx[0], xx[1])
        self.eI += eI

        return I


#It only work for odd number of points is based one above but works up to machine precision
class FE2_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        #new grid
        xx = np.append(x,[2.0*x[self.N-1] - x[self.N-2], 3.0*x[self.N-1]-2*x[self.N-2]] )
        #self.x = np.append([-x[0],0],xx)
        self.x = xx#np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,1.0)* np.heaviside(x2-x,1.0)
    
    def f(self,x,i):
        R=0.0
        ii =i#+1
        if(i==0):
            #R=self.pulse(x,self.x[0],self.x[1])
            #R=self.pulse(x,self.x[1],self.x[2])
        #    R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])
            R+=(x- self.x[1])*(x- self.x[2])/((self.x[0] -self.x[2])*(self.x[0]-self.x[1]))*np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[2]-x,1.0)
            #self.pulse(x,self.x[0],self.x[3])
            #print(i,R,x,(x- self.x[2])*(x- self.x[3])/((self.x[1] -self.x[3])*(self.x[1]-self.x[2])),np.heaviside(x-self.x[3],1.0)* np.heaviside(self.x[0]-x,1.0))
            return R
        elif(i==self.N-1):
            R += (x- self.x[ii-2])*(x- self.x[ii-1])/((self.x[ii] -self.x[ii-2])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-2],self.x[ii])
            return R
        
        if((ii+1)%2==0):#Even
            R  += (x- self.x[ii-1])*(x- self.x[ii+1])/((self.x[ii] -self.x[ii+1])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-1],self.x[ii+1])
            return R
        else:#odd?
            R += (x- self.x[ii-2])*(x- self.x[ii-1])/((self.x[ii] -self.x[ii-2])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-2],self.x[ii  ])
            R += (x- self.x[ii+1])*(x- self.x[ii+2])/((self.x[ii] -self.x[ii+2])*(self.x[ii] -self.x[ii+1]))*self.pulse(x,self.x[ii],self.x[ii+2])
            return R

    
    def set_up_basis(self,newgrid):
        res = np.empty(newgrid.shape[0])
        for i in range(newgrid.shape[0]):
            res[i] = 0.0
            for j in range(self.N):
                res[i] += self.f(newgrid[i],j)#*self.Norm[j]
        return res
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
    
        
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency 
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        return res


    def ComputeI(self,i,Kernel):
        eps=1e-8
        if(i==0):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,0), self.x[0], self.x[2],epsabs=eps)
            self.eI += eI
            return I
        if(i==self.N):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,self.N-1), self.x[self.N-3], self.x[self.N-1],epsabs=eps)
            self.eI += eI
        ii=i
        if((ii+1)%2==0):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-1], self.x[ii+1],epsabs=eps)
            self.eI += eI
        else:
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-2], self.x[ii+2], epsabs=eps)
            self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        # I need to fix the i=0 case
        ii=i+1
        jj=j+1
        if(ii%2==0):
            xx = (self.x[ii-1], self.x[ii+1])
        else:
            xx = (self.x[ii-2], self.x[ii+2])
        if(jj%2==0):
            yy = (self.x[jj-1], self.x[jj+1])
        else:
            yy = (self.x[jj-2], self.x[jj+2])
        
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), yy[0], yy[1],xx[0], xx[1],epsrel=1e-12)
        self.eI += eI

        return I


def interp(x,q,fe):
    S = 0*x
    for k in range(fe.N):
        S+= fe.f(x,k)*q[k]
    return S


#### MODELS ####
class simple_PDF():
    def __init__(self,a,b,g): 
        self.a=a
        self.b=b
        self.g=g
        self.r = 1.0
        self.F = lambda y: (y**a*(1-y)**b*(1 + g*np.sqrt(y)))/self.r
        self.r,e = integrate.quad(self.F,0.0,1.0)  


def DPDFnormed(x,a,b):
    P=tr.tensor([a,b])
    a,b=P[0],P[1]
    dG_da,dG_db=dNorm(P)
    N=tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
    dP_da=(tr.pow(x,a))*tr.pow(1-x,b)*tr.log(x) *N+dG_da*x**a*(1-x)**b
    dP_db= (tr.pow(x,a))*tr.pow(1-x,b)*tr.log(1-x) *N + dG_db*x**a*(1-x)**b
    return dP_da,dP_db

def Normalization(P):
    a,b=P[0],P[1]
    return tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def dNorm(P):
    a,b=P[0],P[1]
    dG_da= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(a+1))
    dG_db= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(b+1))
    return tr.tensor([dG_da,dG_db])


def PDFn(x,a,b):
    return tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
#x**a*(1-x)**b*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))


def very_simplePDFnormed(x,b):
    return (1-x)**b*tr.exp(gammaln(b+2) - gammaln(b+1))

# Posterior GP V2 with split RBF kernel
# Posterior GP V2 with split RBF kernel
def pseudo_data(nu,a,b,g,da,db,dg,N,ITD="Re",Model="PDF"):

    sa = np.random.normal(a,da,N)
    sb = np.random.normal(b,db,N)
    sg = np.random.normal(g,dg,N)

    D = np.zeros((N,nu.shape[0]))
    Norm=1.0
    for k in range(N):
        for i in range(nu.shape[0]):
            if ITD=="Re":
                F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.cos(nu[i]*y) 
            else:
                F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.sin(nu[i]*y)
            r,e = integrate.quad(F,0.0000001,1.0-0.0000001) 
            D[k,i] = r
            if i==0:
                Norm = r
            D[k,i] = D[k,i]/Norm
    #add additional gaussian noise to break correlations
    NN = np.random.normal(0,1e-2,np.prod(D.shape)).reshape(D.shape)
    return D+NN

def autograd(func,x):
    x_tensor = x.clone().detach()
    x_tensor.requires_grad_()
    y = func(x_tensor)
    y.backward()
    return x_tensor.grad

def DPDFnormed(x,a,b):
    P=tr.tensor([a,b])
    a,b=P[0],P[1]
    dG_da,dG_db=dNorm(P)
    N=tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
    dP_da=(tr.pow(x,a))*tr.pow(1-x,b)*tr.log(x) *N+dG_da*x**a*(1-x)**b
    dP_db= (tr.pow(x,a))*tr.pow(1-x,b)*tr.log(1-x) *N + dG_db*x**a*(1-x)**b
    return dP_da,dP_db

def Normalization(P):
    a,b=P[0],P[1]
    return tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def dNorm(P):
    a,b=P[0],P[1]
    dG_da= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(a+1))
    dG_db= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(b+1))
    return tr.tensor([dG_da,dG_db])


def PDF_N(x,a,b,N):
    return N*tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def PDF_con(x,b):
    a=tr.tensor(0.25)
    return tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def PDF_div(x,b):
    a=tr.tensor(-0.5)
    return tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
#xtensor=tr.tensor(x_grid)
def model(x):
    a=x[0]
    b=x[1]
    xtensor=tr.tensor([0.5])
    return PDF_N(xtensor,a,b)

def g_flat(x,N):
    return x-x+N

def noModel(x,N):
    return x-x+N*0

def PDF(x,a,b,N):
    return N*tr.pow(x,a)*tr.pow(1-x,b)

#### Kernels #####

def KrbfMat(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return (s)*tr.exp(-0.5*((xx - yy)/w)**2)

def KrbfMatxa(x,s,w,a):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return (s)*xx**a*tr.exp(-0.5*((xx - yy)/w)**2)*yy**a

def KrbfMatxab(x,s,w,a,b):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return (s)*xx**a*(1-xx)**b*tr.exp(-0.5*((xx - yy)/w)**2)*yy**a*(1-yy)**b

def rbf_s(x,w):
    s=10.0
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*tr.exp(-0.5*((xx - yy)/w)**2)

def Krbflog(x,s,w,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)

def Krbflogxa(x,s,w,a,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*xx**a*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)*yy**a

def Krbflogxab(x,s,w,a,b,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*xx**a*(1-xx)**b*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)*yy**a*(1-yy)**b

def Krbflog_no_s(x,w,s=5.0,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)

def Krbf_no_s(x,w):
    s=2.0
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)


def Krbf_fast(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    w=10**w
    s=10**s
    return s**2*tr.exp(-0.5*((xx - yy)/w)**2)

def Kpoly(x,s,t,a,b):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*((xx*yy)**a*((1-xx)*(1-yy))**b)/((1-t*xx)*(1-t*yy))

def Kpoly1(x,s,a,b):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*((xx*yy)**a*((1-xx)*(1-yy))**b)/((1-yy*xx))

def log_poly1(x,s1,w1,s,a,b,scale=1.0,sp=0.1,eps=1e-13):
    K2= Krbflog(x,s1,w1,eps) #log # linear
    K1 = Kpoly1(x,s,a,b) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    sC= 1-s
    return s*K1*s.T + sC*K2*sC.T

def log_jac(x,s,t,a,b,s1,w1,scale,sp=0.1,eps=1e-12):
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K2 = KrbfMat(tr.log(x+eps),s1,w1) #log # linear
    K1 = jacobi(x,s,t,a,b) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def l(x,l0,eps=1e-10):
    return l0*(x+eps)

def Kdebbio(x,sig,l0,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))



class splitRBFker():
    def __init__(self,sp,scale=1):
        self.sp =sp
        self.scale = scale
    def KerMat(self,x,s1,w1,s2,w2):
        K2 = KrbfMat(x,s2,w2) # linear
        K1 = KrbfMat(tr.log(x),s1,w1)
        sig = tr.diag(tr.special.expit(self.scale*(x-self.sp)))
        sigC = tr.eye(x.shape[0])-sig
        ##return K1+K2
        return sigC@K2@sigC + sig@K1@sig

def Sig(x,scale,sp=0.1):
    return tr.special.expit(scale*scale*(x-sp))
def transform(s):
    return s.view(s.shape[1],1).repeat(1,s.shape[1])

#  write the last one as a function
def rbf_logrbf(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_logrbf_l(x,w1,w2,s1=1.0,s2=50.0,scale=1.0,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_logrbf_s1(x,s1,w1,s2,w2,scale=1.0,sp=0.1,eps=1e-13):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T


#  write the last one as a function
def rbf_logrbf_s_w(x,s1,s2,scale=1,sp=0.1,eps=1e-14):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,s1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,s2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_debxa(x,s1,w1,s2,w2,a,scale,sp=0.1,eps=1e-11):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbioxa(x,s2,w2,a,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_s1(x,s1,w1,s2,w2,scale=1.0,sp=0.1,eps=1e-13):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_s1_s2(x,w1,w2,scale=1.0,sp=0.1,eps=1e-13):
    s1=1.5
    s2=2.5
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_s_w(x,s1,s2,scale=1,sp=0.1,eps=1e-13):
    K1 = KrbfMat(x,s1,s1) # linear
    K2 = Kdebbio(x,s2,s2,eps=eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def splitRBF1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def l(x,l0,eps=1e-10):
    return l0*(x+eps)


def Kdebbio(x,l0,sig,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return sig*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))

def Kdebbioxa(x,sig,l0,a,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return xx**a*sig*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(2*l(xx,l0,eps)**2+2*l(yy,l0,eps)**2))*yy**a

def Kdebbioxa_no_s(x,l0,a,sig,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return xx**a*sig*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(2*l(xx,l0,eps)**2+2*l(yy,l0,eps)**2))*yy**a

def Kdebbioxb(x,l0,sig,b,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return (1-xx)**b*sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*(1-yy)**b

def KSM_bad(x,s1,l1,m1,s2,l2,m2):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s1*tr.exp(-0.5*((xx - yy)**2/l1**2))*tr.cos(m1*(xx-yy)**2)+s2*tr.exp(-0.5*((xx - yy)**2/l2**2))*tr.cos(m2*(xx-yy)**2)

def KSM(x,s1,l1,m1,s2,l2,m2):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s1*tr.exp(-0.5*((xx - yy)**2/l1**2))*tr.cos(2*tr.pi*m1*(xx-yy))+s2*tr.exp(-0.5*((xx - yy)**2/l2**2))*tr.cos(2*tr.pi*m2*(xx-yy))

def R(z,t):
    return tr.sqrt(1-2*z*t+t*t)

def F(z,t,a,b):
    return 1/(R(z,t)*(1-t+R(z,t))**a*(1+t+R(z,t))**b)

def jacobi(x,s,t,a,b):
   x=x.view(x.shape[0],1)
   y=x.view(1,x.shape[0])
   return (s**2)*(x*y)**a*((1-x)*(1-y))**b* F(2*x-1,t,a,b)* F(2*y-1,t,a,b)

def rbf_logrbf_no_s(x,w1,w2,s1=5.0,s2=10.0,scale=5.0,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_no_s(x,w1,w2,s1=1.0,s2=1.0,scale=1.0,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T


#DERIVATIVES
def Krbf_ds(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return 2*s*tr.exp(-0.5*((xx - yy)/w)**2)
    #return  2*s*tr.exp(-0.5*((x.view(1,x.shape[0]) - x.view(x.shape[0],1))/w)**2)
def Krbf_dw(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)*(xx-yy)**2/((w**3))

def sig_ds(x,scale,sp=0.1):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    return sig*(1-sig)

def Kcom_ds1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    #sigC = 1-sig
    return sig*Krbf_ds(x,s1,w1)*sig.T
def Kcom_dw1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    #sigC = 1-sig
    return sig*Krbf_dw(x,s1,w1)*sig.T
def Kcom_ds2(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-15):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    sigC = 1-sig
    return sigC*Krbf_ds(tr.log(x+eps),s2,w2)*sigC.T
def Kcom_dw2(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    sigC = 1-sig
    return sigC*Krbf_dw(tr.log(x+eps),s2,w2)*sigC.T

def sig_ds(x,scale,sp=0.1):
    return tr.exp(-scale*(x-sp))*(x-sp)*tr.special.expit(scale*(x-sp))**2

def Kcom_ds(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    K2=KrbfMat(tr.log(x+eps),s2,w2)
    K1=KrbfMat(x,s1,w1)
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    ##vectors
    ssx=Sig(xx,scale,sp)
    ssy=Sig(yy,scale,sp)
    #transform into matrix
    sx=transform(ssx)
    sy=transform(ssy.T)

    dssx=sig_ds(xx,scale,sp)
    dssy=sig_ds(yy,scale,sp)
    #transform into matrix
    dsx=transform(dssx)
    dsy=transform(dssy.T)

    F1=((-1+sy.T)*dsx + (sx-1)*dsy.T)*K2
    F2=((dsx)*sy.T + sx*(dsy.T))*K1
    return F1+F2

def R(z,t):
    return 1.0/tr.sqrt(1-2*z*t+t*t)

def jacobi_t(x,s,a,b):
   t=tr.tensor(0.5)
   x=x.view(x.shape[0],1)
   y=x.view(1,x.shape[0])
   return (s**2)*(x*y)**a*((1-x)*(1-y))**b*(R(2*x-1,t)*R(2*y-1,t)*((1-t+R(2*x-1,t))*(1-t+R(2*y-1,t)))**a*((1+t+R(2*x-1,t))*(1+t+R(2*y-1,t)))**b)**(-1)

#set up input data
def preparedata(i,nu,rMj,rMe,rM,x_grid,ITD="Re"):


    #prepare the data
    Nx = x_grid.shape[0]
    #CovD= np.corrcoef(rMj[:,:,i-1].T)#*(rMj[:,:,i-1].T.shape[0]-1)

    Nj = 349 #data points

    """ Np = 6
    Nz = 12
    Nj = 349
    rMj = np.empty([Nj,Np,Nz])"""
    #indices [p,z,
    CovD= np.cov(rMj[:,:,i].T*np.sqrt(Nj-1))#same factor used to plot the data
    CovD=(CovD+CovD.T)/2.0

    #CovD=np.abs(CovD)
    #CovD[CovD<0]=0

    M = rM.T[i]
    eM = rMe.T[i]
    n = nu.T[i]
    regu= 1e-8*np.identity(n.shape[0])
    fe = FE2_Integrator(x_grid)
    # soften the constrants
 
    B0 = fe.set_up_integration(Kernel=lambda x: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end...
    n # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
            lam = 1e-5  #normalization
            lam_c = 1e-6 #x=1 

        elif ITD=="Im":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(n[k]*x))
            #lam = 1e5  #normalization
            lam_c = 1e-6 #x=1 
            

    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        if i<-12:
            Gamma[2:,2:] = np.diag(np.diag(CovD))
        else:
            Gamma[2:,2:] = CovD + 0.01*np.diag(np.diag(CovD))#np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        #Gamma[0,0] = lam
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD +0.01*np.diag(np.diag(CovD))#np.diag(eM)#CovD
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma

#gamma function numpy
import math
def gamma(x):
    return math.gamma(x)
def PDF_np(x,a,b):
    return x**a*(1-x)**b*gamma(a+b+2)/(gamma(a+1)*gamma(b+1))
def mockpdf(xgrid,a,b,da,db,N):

    pdf = np.zeros((xgrid.shape[0],N))
    sa = np.random.normal(a,da,N)
    sb = np.random.normal(b,db,N)
    for i in range(xgrid.shape[0]):
        for j in range(N):
            pdf[i,j] = PDF_np(xgrid[i],sa[j],sb[j])
    return pdf

def pseudo_data1(nu_grid,x_grid,a,b,da,db,ITD="Re"):
    #generate data
    fe=FE2_Integrator(x_grid)
    BB = np.zeros((nu_grid.shape[0],x_grid.shape[0]))
    for k in np.arange(nu_grid.shape[0]):
        if ITD=="Re":
            BB[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_grid[k]*x))
        elif ITD=="Im":
            BB[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_grid[k]*x))
    pdf=mockpdf(x_grid,a,b,da,db,1000)
    return pdf.T @ BB.T

def preparemockdata1(Nnupoints,numax,x_grid,ITD="Re"):
    nu_grid = np.linspace(0,numax,Nnupoints)
    Nx=x_grid.shape[0]
    nu_grid=nu_grid[1:]
    if ITD=="Re":
        if numax==25:
            Reg=1e-11*np.identity(nu_grid.shape[0])
        elif numax==10:
            Reg=1e-12*np.identity(nu_grid.shape[0])
        elif numax==4:
            Reg=1e-13*np.identity(nu_grid.shape[0])
    elif ITD=="Im":
        if numax==25:
            Reg=1e-8*np.identity(nu_grid.shape[0])
        elif numax==10:
            Reg=1e-10*np.identity(nu_grid.shape[0])
        elif numax==4:
            Reg=1e-13*np.identity(nu_grid.shape[0])

    fe = FE2_Integrator(x_grid)

    #generate data
    itd=pseudo_data1(nu_grid,x_grid,-0.2,2.5,0.1,0.5,ITD=ITD)
    M=itd.mean(axis=0)
    CovD= np.cov(itd.T)

    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((nu_grid.shape[0],Nx))
    for k in np.arange(nu_grid.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_grid[k]*x))
            lam = 1e-10   # soften the constrants
            lam_c = 1e-10
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_grid[k]*x))
            lam_c = 1e-10
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        Gamma[2:,2:] = CovD + Reg #np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma

def covfilter(cov,n):
    N=cov.shape[0]
    for i in range(N):
        for j in range(N):
            if np.abs(i-j)>n-1:
                cov[i,j]=0
    return cov

def NNPDFdata(datanu,x_grid,regulator,lamb,ITD="Re",corr=True):
    nu_d_grid = datanu.T[1]
    numax=nu_d_grid.shape[0]
    #print("numax: ",numax)
    Nx=x_grid.shape[0]
    if regulator:
        if ITD=="Re":
            if numax==25:
                Reg=regulator[0]*np.identity(nu_d_grid.shape[0])#1e-7
            elif numax==10:
                Reg=regulator[1]*np.identity(nu_d_grid.shape[0])#1e-9
            elif numax==4:
                Reg=regulator[2]*np.identity(nu_d_grid.shape[0])#1e-10
        elif ITD=="Im":
            if numax==25:
                Reg=regulator[0]*np.identity(nu_d_grid.shape[0])#1e-7
            elif numax==10:
                Reg=regulator[1]*np.identity(nu_d_grid.shape[0])#1e-9
            elif numax==4:
                Reg=regulator[2]*np.identity(nu_d_grid.shape[0])#1e-10
    else:
        Reg=0

    M=datanu.T[2:].mean(axis=0)
    eMnu=datanu.T[2:].std(axis=0)#*np.sqrt(datanu.shape[0])
    CovD=np.cov(datanu.T[2:].T)#*np.sqrt(datanu.shape[0]-1))
    #Symetrize the matrix CovD
    #print("Symetrize the matrix CovD")
    CovD=(CovD+CovD.T)/2.0
    #truncate covariance matrix
    if numax==25:
        #print("Flag1")
        CovD=truncatecov(CovD)
        regu= 1e-9*np.identity(nu_d_grid.shape[0])
    else:
        regu= 1e-4*np.diag(np.diag(CovD))
    #CovD=np.abs(CovD)

    if not corr:
        CovD = np.diag(np.diag(CovD))

    fe = FE2_Integrator(x_grid)
    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = x_grid[-1]#1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((nu_d_grid.shape[0],Nx))
    for k in np.arange(nu_d_grid.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_d_grid[k]*x))
            lam = lamb[0]  # soften the constrants
            lam_c = lamb[1]
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_d_grid[k]*x))
            lam_c = lamb[1]
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        #Gamma[2:,2:] = CovD + Reg#np.diag(eM)#CovD
        if numax>30:
            #print("Flag1")
            Gamma[2:,2:] = np.diag(np.diag(CovD))
        else:
            Gamma[2:,2:] = CovD + regu#np.diag(eM**2)
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        if numax>30:
            Gamma[1:,1:] = np.diag(np.diag(CovD))
        else:
            Gamma[1:,1:] = CovD + regu#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma

#Optimal truncation of the covariance matrix

def truncatecov(cov):
    svd=np.linalg.svd(cov)
    med=np.median(svd[1])
    optimal=med*4/np.sqrt(3)
    svd[1][svd[1]<optimal]=0
    return svd[0] @ np.diag(svd[1]) @ svd[2]


def preparemockdata(Nnupoints,numax,ITD="Re",Nx=256):
    #MOCK data
    #######Generate mock data to test the GP

    numock = np.linspace(0,numax,Nnupoints)
    #create fake data
    if numax==25:
        Reg=1e-11*np.identity(Nnupoints)
    elif numax==10:
        Reg=1e-12*np.identity(Nnupoints)
    elif numax==4:
        Reg=1e-13*np.identity(Nnupoints)
    #a,b,c
    jM = pseudo_data(numock,-0.2,3.0,0.01,0.2,0.2,0.01,1000,ITD=ITD)
    #print(jM.shape)
    M = np.mean(jM,axis=0)
    eM = np.std(jM,axis=0)

    jM = jM[:,1:]
    n = numock[1:]
    M = np.mean(jM,axis=0)
    eM = np.std(jM,axis=0)
    
    
    #print("jM shape: ",jM.shape)

    #CovD = np.corrcoef(jM.T)   

    CovD= np.cov(jM.T)
    CovD=(CovD+CovD.T)/2
    #CovD=CovD**0.5
    #change nans by 0
    #CovD=np.abs(CovD)
    CovD[np.isnan(CovD)]=0

    x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1-1e-12,np.int32(Nx/2))))
    fe = FE2_Integrator(x_grid)

    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
            lam = 1e-10   # soften the constrants
            lam_c = 1e-10
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(n[k]*x))
            lam_c = 1e-10
            
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        Gamma[2:,2:] = CovD + Reg #np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))
    return x_grid,V,Y,Gamma


def arguments(modelname,kernelname,nugget,device,mode,ID,grid,Nx,ITD):
    if modelname==PDF_N.__name__:
        meanf=tr.tensor([-1.0,0.0,0.0])
        sigmaf=tr.tensor([2.0,15.0,15.0])
        configf=tr.tensor([2,2,2])
        mod=(-0.25,3.0,1.0)
        #if ITD=="Re":
        #    mod=(-0.25,3.0,1.0)
        #elif ITD=="Im":
        #    mod=(-0.25,3.0,0.5)
        labf=['α', 'β', 'N']
        modfunc=PDF_N

    elif modelname==g_flat.__name__:#Constant model
        meanf=tr.tensor([0.0])
        sigmaf=tr.tensor([15.0])
        configf=tr.tensor([2.0])
        mod=(1.0,)
        #if mode=="kernel":
        #    mod=(0.0,)
        labf=['N']#=sigma
        modfunc=g_flat

    elif modelname==noModel.__name__:
        meanf=tr.tensor([0.0])
        sigmaf=tr.tensor([15.0])
        configf=tr.tensor([2])
        mod=(0.0,)
        labf=['N']
        modfunc=noModel
    elif modelname==PDF_con.__name__:
        meanf=tr.tensor([0.0])
        sigmaf=tr.tensor([15.0])
        configf=tr.tensor([2.0])
        mod=(3.0,)
        labf=['β']
        modfunc=PDF_con
    
    elif modelname==PDF_div.__name__:
        meanf=tr.tensor([0.0])
        sigmaf=tr.tensor([15.0])
        configf=tr.tensor([2])
        mod=(3.0,)
        labf=['β']
        modfunc=PDF_div

    elif modelname=="PDFc":
        meanf=tr.tensor([-1.0,0.0])
        sigmaf=tr.tensor([1.0,15.0])
        configf=tr.tensor([2,2])
        mod=(0.25,3.0)
        labf=['α', 'β']
        modfunc=PDFn
    
    elif modelname=="PDFd":
        meanf=tr.tensor([0.0,0.0])
        sigmaf=tr.tensor([1.0,15.0])
        configf=tr.tensor([2,2])
        mod=(-0.5,3.0)
        labf=['α', 'β']
        modfunc=PDFn

    elif modelname==PDFn.__name__:
        meanf=tr.tensor([-1.0,0.0])
        sigmaf=tr.tensor([2.0,15.0])
        configf=tr.tensor([2,2])
        mod=(-0.1,1.0)
        labf=['α', 'β']
        modfunc=PDFn


    elif modelname==PDF.__name__:
        meanf=tr.tensor([-1.0,0.0,0.0])
        sigmaf=tr.tensor([2., 15., 15.])#2.0,7.0,20.0])
        configf=tr.tensor([2,2,2])
        if ITD=="Re":
            mod=(-0.25,3.0,2.0)
        elif ITD=="Im":
            mod=(-0.1,3.0,2.0)
        #if mode=="kernel":
        #    mod=(0.0,0.0,0.0)
        labf=['α', 'β', 'N']
        modfunc=PDF

    #select the kernel
    if kernelname==rbf_logrbf.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([5., 1., 5., 1.,  2.]) #ModelC
        #sigmak=tr.tensor([11.0,6.0,11.0,6.0,2.0])
        configk=tr.tensor([2,2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(1.0,np.log(3.5),50.0,np.log(2.0),1.0)
        labk=['σ1','l1','σ2','l2','s']
        kerfunc=rbf_logrbf

    elif re.match( r'^Krbflog_no_sn=(-?\d+\.\d+)$',kernelname):
        res=re.match( r'^Krbflog_no_sn=(-?\d+\.\d+)$',kernelname)
        s1 = float(res.group(1))
        meank=tr.tensor([0.0])
        sigmak=tr.tensor([1.0])
        configk=tr.tensor([2])
        ker=(np.log(2),)
        labk=['l']
        kerfunc=lambda x, w: Krbflog_no_s(x, w, s=s1, eps=1e-13)

    if kernelname==rbf_logrbf_l.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([ 1.5, 1.0]) #ModelC
        #sigmak=tr.tensor([11.0,6.0,11.0,6.0,2.0])
        configk=tr.tensor([2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(np.log(2.5),np.log(2.0))
        labk=['l1','l2']#low x, high x
        kerfunc=rbf_logrbf_l

    if re.match( r'^rbf_logrbf_ln=(-?\d+\.\d+)$',kernelname):
        res=re.match( r'^rbf_logrbf_ln=(-?\d+\.\d+)$',kernelname)
        ss2 = float(res.group(1))
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([ 1.5, 1.0]) #ModelC
        #sigmak=tr.tensor([11.0,6.0,11.0,6.0,2.0])
        configk=tr.tensor([2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(np.log(2.5),np.log(2.0))
        labk=['l1','l2']#low x, high x
        kerfunc=lambda x, w1,w2: rbf_logrbf_l(x, w1,w2, s1=2.0,s2=ss2, eps=1e-13)
        #kerfunc=rbf_logrbf_l

    elif kernelname==rbf_deb.__name__: 
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([15.0,15.0,15.0,15.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2])
        ker=(2.5,0.5,2.5,0.5,1.0)
        labk=['σ1','l1','σ2','l2','s']
        kerfunc=rbf_deb


    elif kernelname==rbf_debxa.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,-1.0,0.0])
        sigmak=tr.tensor([5.0,1.0,5.0,1.0,1.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2])
        if ITD=="Re":
            ker=(1.0,np.log(3.5),10.0,np.log(2.0),-0.25,1.0)
        elif ITD=="Im":
            ker=(1.0,np.log(3.5),10.0,np.log(2.0),-0.25,1.0)
        labk=['σ1','l1','σ2','l2','α','s']
        kerfunc=rbf_debxa

    if kernelname==rbf_logrbf_s1.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20.0,20.0,20.0,20.0])
        configk=tr.tensor([2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(5.0,0.1,5.0,1.0)
        labk=['σ1','l1','σ2','l2']
        kerfunc=rbf_logrbf_s1

    if kernelname==rbf_logrbf_s_w.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([20.0,20.0])
        configk=tr.tensor([2,2])
        ker=(2.0,2.0)
        labk=['σ1','σ2']
        kerfunc=rbf_logrbf_s_w

    if kernelname==rbf_deb_s1.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20.0,20.0,20.0,20.0])
        configk=tr.tensor([2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(2.5,0.5,2.5,0.5)
        labk=['σ1','l1','σ2','l2']
        kerfunc=rbf_deb_s1

    if kernelname==rbf_deb_s1_s2.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(2.5,2.5)
        labk=['l1','l2']
        kerfunc=rbf_deb_s1_s2

    elif kernelname==rbf_deb_s_w.__name__: 
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([11.0,6.0])
        configk=tr.tensor([2,2])
        ker=(5.0,5.1)
        labk=['σ1','σ2']
        kerfunc=rbf_deb_s_w


    elif kernelname==rbf_logrbf_no_s.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        ker=(0.1,0.1)
        labk=['l1','l2']
        kerfunc=rbf_logrbf_no_s

    elif kernelname==KrbfMat.__name__: 
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([5.0,1.0])
        configk=tr.tensor([2,2])
        ker=(2.5,0.3)
        labk=['σ','l']
        kerfunc=KrbfMat

    elif kernelname==KrbfMatxa.__name__: 
        meank=tr.tensor([0.0,0.0,-1.0])
        sigmak=tr.tensor([5.0,1.0,1.0])
        configk=tr.tensor([2,2,2])
        ker=(1.5,0.4,-0.2)
        labk=['σ','l','α']
        kerfunc=KrbfMatxa
    
    elif kernelname==KrbfMatxab.__name__:
        meank=tr.tensor([0.0,0.0,-1.0,0.0])
        sigmak=tr.tensor([5.0,1.0,1.0,10.0])
        configk=tr.tensor([2,2,2,2])
        ker=(1.5,0.4,-0.2,1.0)
        labk=['σ','l','α','β']
        kerfunc=KrbfMatxab
    
    elif kernelname=="KrbfMat1": 
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([5.0,1.0])
        configk=tr.tensor([2,2])
        ker=(2.5,0.3)
        labk=['σ','l']
        kerfunc=KrbfMat

    elif kernelname==rbf_s.__name__:
        meank=tr.tensor([0.0])
        sigmak=tr.tensor([15.0])
        configk=tr.tensor([2])
        ker=(0.5,)
        labk=['l',]
        kerfunc=rbf_s

    elif kernelname==Krbflog.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([5.0,1.0])
        configk=tr.tensor([2,2])
        ker=(4.0,np.log(2.0))
        labk=['σ','l']
        kerfunc=Krbflog

    elif kernelname=="Krbflog1":
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([5.0,1.0])
        configk=tr.tensor([2,2])
        ker=(2.0,0.5)
        labk=['σ','l']
        kerfunc=Krbflog

    elif kernelname==Krbflog_no_s.__name__:
        meank=tr.tensor([0.0])
        sigmak=tr.tensor([1.0])
        configk=tr.tensor([2])
        ker=(np.log(2),)
        labk=['l']
        kerfunc=Krbflog_no_s

        
    elif kernelname==Krbf_no_s.__name__:
        meank=tr.tensor([0.0])
        sigmak=tr.tensor([1.0])
        configk=tr.tensor([2])
        ker=(0.1,)
        labk=['l']
        kerfunc=Krbf_no_s

    elif kernelname==Krbf_fast.__name__:
        meank=tr.tensor([-6.0,6.0])
        sigmak=tr.tensor([12.0,12.0])
        configk=tr.tensor([2,2])
        ker=(0.0,0.0)
        labk=['10^σ','10^l']
        kerfunc=Krbf_fast

    elif kernelname==Kdebbio.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([15.0,15.0])
        configk=tr.tensor([2,2])
        ker=(2.0,0.5)
        labk=['σ','l']
        kerfunc=Kdebbio

    elif re.match( r'^Kdebbioxa_no_sn=(-?\d+\.\d+)$',kernelname):
        res=re.match( r'^Kdebbioxa_no_sn=(-?\d+\.\d+)$',kernelname)
        s1 = float(res.group(1))
        meank=tr.tensor([0.0,-1.0])
        sigmak=tr.tensor([1.0,1.0])
        configk=tr.tensor([2,2])
        ker=(np.log(2.0),-0.25)
        labk=['l','α']
        kerfunc=lambda x, w,a: Kdebbioxa_no_s(x, w, a,sig=s1, eps=1e-13)

    elif kernelname==Kdebbioxa.__name__:
        meank=tr.tensor([0.0,0.0,-1.0])
        sigmak=tr.tensor([5.0,1.0,1.0])
        configk=tr.tensor([2,2,2])
        ker=(4.0,np.log(2.0),-0.25)
        labk=['σ','l','α']
        kerfunc=Kdebbioxa

    elif kernelname==Krbflogxa.__name__:
        meank=tr.tensor([0.0,0.0,-1.0])
        sigmak=tr.tensor([5.0,1.0,1.0])
        configk=tr.tensor([2,2,2])
        ker=(2.0,0.5,-0.2)
        labk=['σ','l','α']
        kerfunc=Krbflogxa

    elif kernelname==Krbflogxab.__name__:
        meank=tr.tensor([0.0,0.0,-1.0,0.0])
        sigmak=tr.tensor([5.0,1.0,1.0,10.0])
        configk=tr.tensor([2,2,2,2])
        ker=(2.0,0.5,-0.2,1.0)
        labk=['σ','l','α','β']
        kerfunc=Krbflogxab

    elif kernelname==Kdebbioxb.__name__:
        meank=tr.tensor([0.0,0.0,0.0])
        sigmak=tr.tensor([5.0,1.0,10.0])
        configk=tr.tensor([2,2,2])
        ker=(3.0,2.1,3.0)
        labk=['σ','l','β']
        kerfunc=Kdebbioxb

    elif kernelname==KSM.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([5.0,1.0,3.0,5.0,1.0,3.0])
        configk=tr.tensor([2,2,2,2,2,2])
        ker=(1.0,0.5,1.5,1.0,0.5,1.5)
        labk=['σ1','l1','τ1','σ2','l2','τ2']
        kerfunc=KSM
    
    elif kernelname==Kpoly.__name__:
        meank=tr.tensor([0.0,-1.0,0.0,0.0])
        sigmak=tr.tensor([10.0,2.0,10.0,10.0])
        configk=tr.tensor([2,2,2,2])
        ker=(4.0,0.5,-0.1,3.0)
        labk=['σ','t','a','b']
        kerfunc=Kpoly
    
    elif kernelname==Kpoly1.__name__:
        meank=tr.tensor([0.0,-1.0,0.0])
        sigmak=tr.tensor([15.0,2.0,15.0])
        configk=tr.tensor([2,2,2])
        ker=(4.0,-0.1,3.0)
        labk=['σ','a','b']
        kerfunc=Kpoly1

    elif kernelname==log_poly1.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([5.0,1.0,15.0,15.0,15.0])
        configk=tr.tensor([2,2,2,2,2])
        ker=(1.5,0.5,1.5,1.0,1.0)
        labk=['σ1','l1','σ','a','b']
        kerfunc=log_poly1

    elif kernelname==log_jac.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([5.0,1.0,10.0,10.0,5.0,1.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2,2])
        ker=(2.5,0.5,2.0,1.0,9.0,1.0,1.0)
        labk=['σ1','l1','σ2','l2','a','b','s']
        kerfunc=log_jac
    elif kernelname==jacobi.__name__:
        meank=tr.tensor([0.0,-1.0,0.0,0.0])
        sigmak=tr.tensor([15.0,2.0,15.0,15.0])
        configk=tr.tensor([2,2,2,2])
        ker=(2.5,0.1,1.0,1.0)
        labk=['σ','t','a','b']
        kerfunc=jacobi

    elif kernelname==jacobi_t.__name__:
        meank=tr.tensor([0.0,0.0,0.0])
        sigmak=tr.tensor([15.0,15.0,15.0])
        configk=tr.tensor([2,2,2])
        ker=(2.5,1.0,1.0)
        labk=['σ','a','b']
        kerfunc=jacobi_t

    if nugget=="yes":
        meank=tr.cat((meank,tr.tensor([0.0])))
        sigmak=tr.cat((sigmak,tr.tensor([10.0])))
        configk=tr.cat((configk,tr.tensor([2])))
        ker=ker+(1.0,)
        labk=labk+['σ']

    #stack the spec model and kernel
    if mode=="mean":
        mean=meanf
        sigma=sigmaf
        config=configf
        lab=labf
        
    elif mode=="kernel":
        mean=meank
        sigma=sigmak
        config=configk
        lab=labk
    elif mode=="all":
        mean=tr.cat((meanf,meank))
        sigma=tr.cat((sigmaf,sigmak))
        config=tr.cat((configf,configk))
        lab=labf+labk

    x_grid=generategrid(Nx,grid)

    return mean,sigma,config,mod,ker,modfunc,kerfunc,device,mode,ID,x_grid,lab

#Following grids specified in the draft
def generategrid(Nx,grid):
    if grid=="lin":
        x_grid = np.linspace(0.0+1e-6,1-1e-6,np.int32(Nx+1))
    elif grid=="log_lin":
        x_grid=np.concatenate((np.linspace(5e-9,1,1) ,np.logspace(-8,-1,np.int32(Nx/2)), np.linspace( 0.1+1e-4 ,1-1e-8,np.int32(Nx/2))))
    return x_grid


#select the model and kernel
def Modeldef(ITD,modelname,kernelname,nugget,device,mode,ID,test,grid,Nx,lambdas):
    fits_comb=[]
    mean,sigma,config,mod,ker,modfunc,kerfunc,device,mode,ID,x_grid,lab = arguments(modelname,kernelname,nugget,device,mode,ID,grid,Nx,ITD)
    nu,rMj,rMe,rM = get_data(ITD)
    now = datetime.datetime.now()
    print("#################Define the model###########################")
    print ("Current date and time :", now.strftime("%Y-%m-%d %H:%M:%S"))
    print("GP specifications \n Sampling or training: "+mode+"\n model: "+modelname+"\n kernel: "+kernelname+" nugget: "+ nugget+"\n Ioffe time Distribution: "+ITD+"(M)",
          "\n mean =",mean,"\n sigma =",sigma,"\n prior dist =",config,"\n model init =",mod,"\n kernel init =",ker,"\n device =",device,"\n mode =",mode,"\n ID =",ID)
    #print("0=gaussian, 1=lognormal, 2=expbeta")
    if not (test in ["mock","NNPDF"]):
        for i in range(0,12):
            x_gri0,V0,Y0,Gamma0 = preparedata(i,nu,rMj,rMe,rM,x_grid,ITD=ITD)
            myGP0= GaussianProcess(x_gri0,V0,Y0,Gamma0,f"z={i+1}a",device=device,ITD=ITD,kernelname=kernelname,modelname=modelname,labels=lab,gridname=grid,nugget=nugget,Pd=modfunc, Ker=kerfunc,Pd_args=mod,Ker_args=ker)
            myGP0.prior2ndlevel(mode,0.99,mean=mean,sigma=sigma,prior_mode=config)
            fits_comb.append(myGP0)
            #print(fits_comb[i].name, "done")
    if ITD=="Re" and test=="mock":
        numax=[4,10,25]
        for j in range(0,3):
            x_gri0,V0,Y0,Gamma0 = preparemockdata1(numax[j]+1,numax[j],x_grid,ITD)
            myGP0= GaussianProcess(x_gri0,V0,Y0,Gamma0,f"z=mock({numax[j]})",device=device,ITD=ITD,kernelname=kernelname,modelname=modelname,labels=lab,gridname=grid,nugget=nugget,Pd=modfunc, Ker=kerfunc,Pd_args=mod,Ker_args=ker)
            myGP0.prior2ndlevel(mode,0.99,mean=mean,sigma=sigma,prior_mode=config)
            fits_comb.append(myGP0)
            #print(fits_comb[-1].name, "done")
    elif test=="NNPDF":
        if ITD=="Re":
            MMM='real'
        elif ITD=="Im":
            MMM='imag'
        for i in [4,10,25]:
            datanu4 = np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_itd_'+MMM+'_numax'+str(i)+'.dat',dtype=np.float64)
            x_gri0,V0,Y0,Gamma0 = NNPDFdata(datanu4,x_grid,[1e-11,1e-12,1e-13],lambdas,ITD)
            #if kernelname=="Krbflog_no_s" and i==25:
            #    kerfunc=lambda x, w: Krbflog_no_s(x, w, s=3.0, eps=1e-13)
            myGP0= GaussianProcess(x_gri0,V0,Y0,Gamma0,f"z=NNPDF({i})",device=device,ITD=ITD,kernelname=kernelname,modelname=modelname,labels=lab,gridname=grid,nugget=nugget,Pd=modfunc, Ker=kerfunc,Pd_args=mod,Ker_args=ker)
            myGP0.prior2ndlevel(mode,0.99,mean=mean,sigma=sigma,prior_mode=config)
            fits_comb.append(myGP0)
            print(fits_comb[-1].name, "done")
    return fits_comb