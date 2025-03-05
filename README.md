# Gaussian process

#### Useful commands

If you want to run on run.csh(wm) or (jlab), use:

```bash
sbatch run1.csh PDF rbf_logrbf Re 10 100000 all log_lin 13  # run1.csh needs 8 arguments, last argument goes from [0-14]
sbatch run.csh PDF rbf_logrbf Im 10 100000 all lin     # or 7 if you want to run hmc collins+NNPDF data
```
Check all the available kernels and prior set ups in functions.py

The arguments of run.csh/setjlab.sh goes as follow:

(PDF model) (kernel) (ITD) (number of parallel chains for HMC) (# total samples) (which parameters are going to be sampled) (grid preference)
