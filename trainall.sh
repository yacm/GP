#!/bin/sh
# Ensure the script is executed with exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Runing all with 2 arguments"
    echo "Usage: $0 <model> <kernel>"
    exit 1
fi

# Assigning arguments
model=$1  # Example: PDF
kernel=$2  # Example: Krbflog
grid=$3  # Example: all


# Submitting jobs for all components
sbatch train.csh $model $kernel Re 1 1902 all $grid
sleep 5
sbatch train.csh $model $kernel Im 1 1902 all $grid
sleep 5
sbatch train.csh $model $kernel Re 1 1902 kernel $grid
sleep 5
sbatch train.csh $model $kernel Im 1 1902 kernel $grid
sleep 5
sbatch train.csh $model $kernel Re 1 1902 mean $grid
sleep 5
sbatch train.csh $model $kernel Im 1 1902 mean $grid
#sleep 5
#sbatch train.csh $model $kernel Re 1 1902 all lin
#sleep 5
#sbatch train.csh $model $kernel Im 1 1902 all lin
#sleep 5
#sbatch train.csh $model $kernel Re 1 1902 kernel lin
#sleep 5
#sbatch train.csh $model $kernel Im 1 1902 kernel lin
#sleep 5
#sbatch train.csh $model $kernel Re 1 1902 mean lin
#sleep 5
#sbatch train.csh $model $kernel Im 1 1902 mean lin
#sleep 5
