import torch as tr

if tr.backends.mps.is_available():
    device = tr.device("mps")

elif tr.cuda.is_available():
    device = tr.device("cuda")

else:
    device = tr.device("cpu")

from GP import *
from functions import *

import numpy as np
#import matplotlib.pyplot as plt
import argparse 
import scipy.integrate as integrate
import h5py as h5

# import all packages and set plots to be embedded inline
from scipy.optimize import minimize 
import datetime


print("Device:", device)

parser = argparse.ArgumentParser(description='Gaussian Process arguments')
parser.add_argument('--i', type=int, help='data set to analize 0-11 (mock-data=12)')
parser.add_argument('--Nsamples', type=int, help='number of samples')
parser.add_argument('--burn', type=int, default=0, help='burn-in period')
parser.add_argument('--L', type=int, default=100, help='number of leapfrog steps')
parser.add_argument('--eps', type=float, default=1.0/1000, help='step size')
parser.add_argument('--ITD',type=str,default="Re", help='Real or imaginary part of the data')
parser.add_argument('--mean',type=str,default="simplePDFnormed", help='Prior mean model')
parser.add_argument('--ker',type=str,default="rbf_logrbf", help='Kernel model')
parser.add_argument('--mode',type=str,default="all", help='sampling or training over this parameters(kernel, mean, all)')
parser.add_argument('--IDslurm', type=str, default='', help='ID where the job is runing')
parser.add_argument('--grid',type=str,default='lin',help='linear(lin) or log/lin')
parser.add_argument('--Nx',type=int,default=256,help='number of points in Finite elements integration')


args = parser.parse_args()

print(args)
i=args.i
Nsamples=args.Nsamples
burn=args.burn
L=args.L
eps=args.eps
ITD = args.ITD
modelname=args.mean
kernelmodel=args.ker
mode=args.mode
IDslurm=args.IDslurm
grid=args.grid
Nx=args.Nx


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

#from tensor to list
def tensor2list(tensor):
    return [tensor[i].item() for i in range(tensor.shape[0])]

#Nx=256
#x_grid = np.concatenate((np.linspace(1e-10,1,1),np.linspace(0.0+1e-4,1-1e-6,np.int32(Nx))))
#x_grid = np.concatenate((np.logspace(-8,-1,np.int32(Nx/2)),np.linspace(0.1+1e-6,1-1e-6,np.int32(Nx/2))))

modelname=args.mean
kernelname=args.ker
nugget="yes"
if mode=="mean":
    nugget="no"
 
test="NNPDF"
device="cpu"


mean,sigma,config,mod,ker,modfunc,kerfunc,device,mode,IDslurm,x_grid,lab=arguments(modelname,kernelname,nugget,device,mode,IDslurm,grid,Nx)
momentum=tr.ones_like(mean)

now = datetime.datetime.now()
print ("Current date and time :", now.strftime("%Y-%m-%d %H:%M:%S"))
print("GP specifications \n Sampling or training: "+mode+"\n model: "+modelname+"\n kernel: "+kernelname+" nugget: "+ nugget+"\n Ioffe time Distribution: "+ITD+"(M)","\n mean =",mean,"\n sigma =",sigma,"\n prior dist =",config,"\n model init =",mod,"\n kernel init =",ker,"\n momentum init =",momentum,"\n device =",device,"\n mode =",mode,"\n SLURM_ID =",IDslurm)
print("#################Define the model###########################")
fits_comb=[]
#print("0=gaussian, 1=lognormal, 2=expbeta")
fits_comb=Modeldef(ITD,modelname,kernelname,nugget,device,mode,IDslurm,test,grid,Nx)


#train antimode so that we are sampling fixing on the optimal parameters
if mode=="mean":
    Ntrain=1000
    function="evidence"
    lr=1e-3
    i=args.i

    fits_comb[i].train(Ntrain,lr=lr,mode="kernel",function=function)
elif mode=="kernel":
    Ntrain=1000
    function="evidence"
    lr=1e-3
    i=args.i

    fits_comb[i].train(Ntrain,lr=lr,mode="mean",function=function)
else:
    print("Training all parameters")

Ntrain=1000
function="nlp"
i=args.i

lr=1e-3
for i in reversed(range(0,15)):
    if i in [111]:
        fits_comb[i].train(Ntrain,lr=lr*10,mode=mode,function=function)
    elif i in [143]:
        fits_comb[i].train(Ntrain,lr=lr,mode=mode,function=function)
    else:
        fits_comb[i].train(Ntrain,lr=lr*10,mode=mode,function=function)
    print(tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args  + (fits_comb[i].sig,)))
#end=time.time()
#print("time",end-start)
#look for nans in the parameters
i=args.i
def nans(tup):
    for i in range(len(tup)):
        if tr.isnan(tup[i]):
            return True
    return False

i=args.i


if nans(fits_comb[i].ker_args):
    fits_comb[i].hyperparametersvalues(set="original")

print(fits_comb[i].ker_args)


#absolute value of ker args
"""fits_comb[i].ker_args = tuple(tr.abs(x) for x in fits_comb[i].ker_args)

print(tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args  + (fits_comb[i].sig,)))
"""
##sample all the parameters
momentum=tr.ones_like(mean)
#sigma=tr.tensor([3.0,5.0,6.0,5.0,6.0,5.0,5.5,2.0])
samplers=[]

if ITD=="Re":
    numb=15
else:
    numb=15
for i in range(0,numb):
    #if i<5:

    GPsampler=HMC_sampler(fits_comb[i].nlogpost2levelpdf,device="cpu",diagonal=1.0*momentum,grad=None)
    #rand=tr.rand(7)*5
    
    #GPsampler.q0=tr.tensor(fits_comb[i].ker_args  + (fits_comb[i].sig,)).to("cpu")
    if mode=="kernel":
        GPsampler.q0=tr.tensor(fits_comb[i].ker_args).to("cpu")
        if nugget=="yes":
            GPsampler.q0=tr.tensor(fits_comb[i].ker_args + (fits_comb[i].sig,)).to("cpu")
    elif mode=="mean":
        GPsampler.q0=tr.tensor(fits_comb[i].pd_args).to("cpu")
    elif nugget=="yes":
        GPsampler.q0=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args + (fits_comb[i].sig,)).to("cpu")
    else:
        GPsampler.q0=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args).to("cpu")
    #print(GPsampler.q0)
    #GPsampler.q0=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args).to("cpu")+0.02#+ (fits_comb[i].sig,))
    samplers.append(GPsampler)
    print(fits_comb[i].name,"sampler done")

i=args.i
Nsamples=args.Nsamples
burn=args.burn
L=args.L
eps=args.eps

traceq,tracep,traceH=samplers[i].sample(samplers[i].q0,Nsamples,eps,L)#,update=1)
tr.save(traceq,'%s_%s(%s+%s)/K%s(%s)%s.pt' %(modelname,kernelname,mode,grid,ITD,fits_comb[i].name,IDslurm))

print("#################Sampling done###########################")
now = datetime.datetime.now()
print ("Current date and time :", now.strftime("%Y-%m-%d %H:%M:%S"))
