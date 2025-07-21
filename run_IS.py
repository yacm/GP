#!/usr/bin/env python3

import sys
print("Starting GP", flush=True)

import torch as tr
if tr.backends.mps.is_available():
    device = tr.device("mps")

elif tr.cuda.is_available():
    device = tr.device("cuda")

else:
    device = tr.device("cpu")

from GP import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

print(f"Using device: {device}", flush=True)

import scipy.integrate as integrate
#from orthogonal_poly import legendre_01

from torch.autograd.functional import hessian

import scipy.integrate as integrate

import h5py as h5

# import all packages and set plots to be embedded inline
import numpy as np 
import datetime
from scipy.optimize import minimize 
from scipy import special 
from scipy.optimize import Bounds 
from scipy.linalg import cho_solve 
#from pyDOE import lhs 
import time
#load all the Models and Kernels
from functions import *
import argparse 
print("Checkpoint 2: imports done", flush=True)

parser = argparse.ArgumentParser(description='Gaussian Process arguments')
parser.add_argument('--mean',type=str,default="PDF", help='Prior mean model')
parser.add_argument('--ker',type=str,default="rbf_logrbf", help='Kernel model')
parser.add_argument('--mode',type=str,default="all", help='Important sampling or training over this parameters(kernel, mean, all)')
parser.add_argument('--grid',type=str,default='log_lin',help='linear(lin) or log/lin')

import sys
print("sys.argv:", sys.argv, flush=True)

args = parser.parse_args()
models=args.mean
kernels=args.ker
modes=args.mode
grids=args.grid

print(f"Using mean model: {models}, kernel model: {kernels}, mode: {modes}, grid: {grids}", flush=True)

nugget="no" #only if you want to do another regularization for our GP
device="cpu"
#mode="all"
#grid="lin"
#grid2="log_lin"
test="NNPDF"
ID=12
Nx=128*2

lambdass=[1e-5,1e-6]


fits_Re=Modeldef("Re",models,kernels,nugget,device,modes,ID,test,grids,Nx,lambdass)
fits_Im=Modeldef("Im",models,kernels,nugget,device,modes,ID,test,grids,Nx,lambdass)
#flush
print("Checkpoint 3: Model definitions done", flush=True)

def trainmod1(fits):

    Ntrain=1000
    lr=1e-3
    lik="nlp"
    if fits[0].mode=="all":
        train_model(fits,Ntrain,lik,lr,"all")
    elif fits[0].mode=="kernel":
        if fits[0].modelname in ["PDFd","PDFc"]: #rbf_deb
            train_model(fits,Ntrain,lik,lr,"kernel")
        else:
            train_model(fits,Ntrain,"evidence",lr,"mean")
            train_model(fits,Ntrain,lik,lr,"kernel")
    elif fits[0].mode=="mean":
        if fits[0].kernelname in ["KrbfMat2","Krbflog_no_s"]: #rbf_deb
            train_model(fits,1,"evidence",1e-4,"kernel")
            train_model(fits,Ntrain,lik,lr,"mean")
        else:
            train_model(fits,Ntrain,"evidence",lr,"kernel")     ###criteria p=30%
            train_model(fits,Ntrain,lik,lr,"mean")


def trainmod(fits):
    Ntrain=5000
    lr=5e-4
    lik="nlp"

    if fits[0].mode=="all":
        train_model(fits,Ntrain,lik,lr,"all")
    elif fits[0].mode=="kernel":
        if fits[0].modelname in ["PDFd","PDFc"]: #rbf_deb
            train_model(fits,Ntrain,lik,lr,"kernel")
        else:
            #train_model(fits,Ntrain,"evidence",lr,"mean")
            train_model(fits,Ntrain,lik,lr,"kernel")
    elif fits[0].mode=="mean":
        if fits[0].kernelname in ["rbf_logrbf_l","Kdebbioxa","Krbflog"] or re.match( r'^Krbflog_no_sn=(-?\d+\.\d+)$', fits[0].kernelname): #rbf_deb
            #train_model(fits,Ntrain,"evidence",lr,"kernel")
            train_model(fits,Ntrain,lik,lr,"mean")
        else:
            #train_model(fits,Ntrain,"evidence",lr,"kernel")     ###criteria p=30%
            train_model(fits,Ntrain,lik,lr,"mean")


nn = np.linspace(0,100,128)
x_grid=generategrid(Nx,grids)
Nx=x_grid.shape[0]
iB_Re = np.zeros((nn.shape[0],Nx))
iB_Im = np.zeros((nn.shape[0],Nx)) 
fe_log=FE2_Integrator(x_grid)
for k in range(nn.shape[0]):
    iB_Re[k,:] = fe_log.set_up_integration(Kernel= lambda x : np.cos(nn[k]*x))
    iB_Im[k,:] = fe_log.set_up_integration(Kernel= lambda x : np.sin(nn[k]*x))

def nans(tup):
    for i in range(len(tup)):
        if tr.isnan(tup[i]):
            return True
    return False

def save_min(fits):
    for ll in range(len(fits)):
        try:
            minpoint=tr.tensor(fits[ll].pd_args +fits[ll].ker_args)

            if nans(minpoint):
                print("Nans in the parameters")
            else:
                print("Posterior 2nd level minimized")
            # Ensure the directory exists
            path=f"{fits[ll].modelname}_{fits[ll].kernelname}({fits[ll].mode}+{fits[ll].gridname})/min_is/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tr.save(minpoint,'%s_%s(%s+%s)/min_is/K%s(%s)(train).pt' %(fits[ll].modelname,fits[ll].kernelname,fits[ll].mode,fits[ll].gridname,fits[ll].ITD,fits[ll].name))
        except Exception as e:
            print(f"Error in saving minpoint for {fits[ll].ITD} {fits[ll].name}, {fits[ll].modelname}, {fits[ll].kernelname} {fits[ll].mode} {fits[ll].gridname}: {e}", flush=True)
            continue

def criteria_single_model(fits_comb,data):

    lista_12=[]
    lista_13=[]
    lista_14=[]
    ii=0
    p=0.3
    for gps in fits_comb:
        #ii is the index of gps in fits_comb
        ii=fits_comb.index(gps)
        #ii=14
        numax=-1
        #gps=g_flat_Krbflog_Im_mean_log
        DeltaM=tr.diag(gps.Gamma)[numax]**0.5
        absM=tr.abs(gps.Y[numax])#/2
        DeltaM_th=tr.max(tr.diag(data[3][ii])**0.5)
        if DeltaM_th>=tr.max((1+p)*DeltaM,p*absM):
            print(f"{gps.modelname}_{gps.kernelname}_{gps.ITD}_{gps.mode}_{gps.gridname}")#,DeltaM_th,(1+p)*DeltaM,p*absM)
            #fits_comb, data, modelname, kernelname, ITD,grid= gps
            if gps.name=="z=NNPDF(4)":
                lista_12.append(f"{gps.modelname}_{gps.kernelname}_{gps.ITD}_{gps.mode}_{gps.gridname}")
                print(f"z=NNPDF(4) passed")
                #save a file calles pass_z=NNPDF(4).pt
                tr.save(tr.tensor([1]), "%s_%s(%s+%s)/pass_%s_%s.pt" %(gps.modelname,gps.kernelname,gps.mode,gps.gridname,gps.ITD,gps.name))
            elif gps.name=="z=NNPDF(10)":
                lista_13.append(f"{gps.modelname}_{gps.kernelname}_{gps.ITD}_{gps.mode}_{gps.gridname}")
                print(f"z=NNPDF(10) passed")
                #save a file calles pass_z=NNPDF(10).pt
                tr.save(tr.tensor([1]), "%s_%s(%s+%s)/pass_%s_%s.pt" %(gps.modelname,gps.kernelname,gps.mode,gps.gridname,gps.ITD,gps.name))
            elif gps.name=="z=NNPDF(25)":
                lista_14.append(f"{gps.modelname}_{gps.kernelname}_{gps.ITD}_{gps.mode}_{gps.gridname}")
                print(f"z=NNPDF(25) passed")
                #save a file calles pass_z=NNPDF(25).pt
                tr.save(tr.tensor([1]), "%s_%s(%s+%s)/pass_%s_%s.pt" %(gps.modelname,gps.kernelname,gps.mode,gps.gridname,gps.ITD,gps.name))
    #print("Total number of models with a posterior error greater than the prior error: ",mod,"/", len(modelss))
    return lista_12, lista_13, lista_14

def save_mode(fits):
    for ll in range(len(fits)):
        try:
            mode, counts = compute_mode_per_column(fits[ll].trace_is, decimals=2)
            if fits[ll].mode=="mean":
                mode_t=tr.cat([mode, tr.tensor(fits[ll].ker_args)])
            elif fits[ll].mode=="kernel":
                mode_t=tr.cat([tr.tensor(fits[ll].pd_args), mode])
            else:
                mode_t=mode
            if nans(mode):
                print("Nans in the parameters")
            else:
                print("Posterior 2nd level mode")
            # Ensure the directory exists
            path=f"{fits[ll].modelname}_{fits[ll].kernelname}({fits[ll].mode}+{fits[ll].gridname})/mode_is/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tr.save(mode_t,'%s_%s(%s+%s)/mode_is/K%s(%s)(train).pt' %(fits[ll].modelname,fits[ll].kernelname,fits[ll].mode,fits[ll].gridname,fits[ll].ITD,fits[ll].name))
        except Exception as e:
            print(f"Error in saving mode for {fits[ll].ITD} {fits[ll].name}, {fits[ll].modelname}, {fits[ll].kernelname} {fits[ll].mode} {fits[ll].gridname}: {e}", flush=True)
            continue


kerlist=["KrbfMat_1","Krbflog_1"]#,"rbf_deb"]
if fits_Re[0].mode=="mean":
    if fits_Re[0].kernelname in kerlist:
        fits_Re=redefine_kernel(fits_Re,iB_Re,0.3)#30% criteria
    if fits_Im[0].kernelname in kerlist:
        fits_Im=redefine_kernel(fits_Im,iB_Im,0.3)

#chekpoint
print("Checkpoint 4: Training models", flush=True)

trainmod(fits_Re)
print("Checkpoint 4.1: Training Re model done", flush=True)
trainmod(fits_Im)
print("Checkpoint 4.2: Training Im model done", flush=True)




save_min(fits_Re)
print("Checkpoint 4.3: Saving min points for Re done", flush=True)
save_min(fits_Im)
print("Checkpoint 4.4: Saving min points for Im done", flush=True)



#checkpoint
print("Checkpoint 5: Integration setup done", flush=True)

for fits in [fits_Re,fits_Im]:
    for ll in range(len(fits)):
        try:
            fits[ll].mgauss_IS()
        except Exception as e:
            print(f"Error in model averaging important sampling for {fits[ll].ITD} {fits[ll].name}, {fits[ll].modelname}, {fits[ll].kernelname} {fits[ll].mode} {fits[ll].gridname}: {e}", flush=True)
            continue

#checkpoint
print("Checkpoint 6: Model averaging gaussian def done", flush=True)
lista=[12,13,14]
nn=tr.tensor(nn)
Nsamp=5000

data_Re = Modelaveraging_importantsampling_gauss(fits_Re,nn,Nsamp,iB_Re,lista)
#checkpoint
criteria_single_model(fits_Re,data_Re)
print("Checkpoint 7: Model averaging important sampling for Re done", flush=True)
data_Im = Modelaveraging_importantsampling_gauss(fits_Im,nn,Nsamp,iB_Im,lista)
#checkpoint
criteria_single_model(fits_Im,data_Im)
print("Checkpoint 8: Model averaging important sampling for Im done", flush=True)


save_min(fits_Re)
print("Checkpoint 8.1: Saving mode for Re done", flush=True)
save_min(fits_Im)
print("Checkpoint 8.2: Saving mode for Im done", flush=True)

save_data(fits_Re,data_Re)
#checkpoint
print("Checkpoint 9: Saving data for Re done", flush=True)
save_data(fits_Im,data_Im)
#checkpoint
print("Checkpoint 10: Saving data for Im done", flush=True)


for ll in range(len(fits_Re)):
    try:
        tr.save(fits_Re[ll].meanE,'%s_%s(%s+%s)/data_%s/meanE_%s.pt' %(fits_Re[ll].modelname,fits_Re[ll].kernelname,fits_Re[ll].mode,fits_Re[ll].gridname,fits_Re[ll].ITD,fits_Re[ll].name))
        tr.save(fits_Im[ll].meanE,'%s_%s(%s+%s)/data_%s/meanE_%s.pt' %(fits_Im[ll].modelname,fits_Im[ll].kernelname,fits_Im[ll].mode,fits_Im[ll].gridname,fits_Im[ll].ITD,fits_Im[ll].name))
    except Exception as e:
        print(f"Error in saving meanE for {fits_Re[ll].ITD} {fits_Re[ll].name}, {fits_Re[ll].modelname}, {fits_Re[ll].kernelname} {fits_Re[ll].mode} {fits_Re[ll].gridname}: {e}", flush=True)
        continue
print("Checkpoint 11: Saving meanE done", flush=True)

print("All processes completed successfully", flush=True)

