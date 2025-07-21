import torch as tr
print("GP started")
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
#print current directory 
import os
print(os.getcwd())


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



#from tensor to list
def tensor2list(tensor):
    return [tensor[i].item() for i in range(tensor.shape[0])]
def nans(tup):
    for i in range(len(tup)):
        if tr.isnan(tup[i]):
            return True
    return False

modelname=args.mean
kernelname=args.ker
nugget="no"
if mode=="mean":
    nugget="no"
 
test="NNPDF"
device="cpu"


mean,sigma,config,mod,ker,modfunc,kerfunc,device,mode,IDslurm,x_grid,lab=arguments(modelname,kernelname,nugget,device,mode,IDslurm,grid,Nx)
momentum=tr.ones_like(mean)

now = datetime.datetime.now()
#print ("Current date and time :", now.strftime("%Y-%m-%d %H:%M:%S"))
#print("GP specifications \n Sampling or training: "+mode+"\n model: "+modelname+"\n kernel: "+kernelname+" nugget: "+ nugget+"\n Ioffe time Distribution: "+ITD+"(M)","\n mean =",mean,"\n sigma =",sigma,"\n prior dist =",config,"\n model init =",mod,"\n kernel init =",ker,"\n momentum init =",momentum,"\n device =",device,"\n mode =",mode,"\n SLURM_ID =",IDslurm)
#print("#################Define the model###########################")
fits_comb=[]
#print("0=gaussian, 1=lognormal, 2=expbeta")
fits_comb=Modeldef(ITD,modelname,kernelname,nugget,device,mode,IDslurm,test,grid,Nx)
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

lr=1e-4
for i in reversed(range(0,15)):
    if i in [111]:
        fits_comb[i].train(Ntrain,lr=lr*10,mode=mode,function=function)
    elif i in [143]:
        fits_comb[i].train(Ntrain,lr=lr,mode=mode,function=function)
    else:
        fits_comb[i].train(Ntrain,lr=lr*10,mode=mode,function=function)
    print(tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args  + (fits_comb[i].sig,)))


i=args.i
minpoint=tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args)

if nans(minpoint):
    print("Nans in the parameters")
else:
    print("Posterior 2nd level minimized")

tr.save(minpoint,'%s_%s(%s+%s)/min/K%s(%s)(train).pt' %(modelname,kernelname,mode,grid,ITD,fits_comb[i].name))