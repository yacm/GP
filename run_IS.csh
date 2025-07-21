#!/bin/sh

#SBATCH --job-name=set_up
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/run1.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/run1.log


if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <KERNEL_NAME>"
    exit 1
fi

if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm01,fm04,fm08,fm24"
else
    EXCLUDE=""
fi

models=($1)
modes=($2)
grids=($3)
kernels=($4)

for kernel in ${kernels[@]}; do
  for model in ${models[@]}; do
    for mode in ${modes[@]}; do
      for grid in ${grids[@]}; do

        mkdir -p "${model}_${kernel}(${mode}+${grid})"
        SLURM_SCRIPT="${model}_${kernel}(${mode}+${grid})/job_script.slurm"
        

cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=${model}_${kernel}(${mode}+${grid})
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/${model}_${kernel}(${mode}+${grid})/specs_data.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/${model}_${kernel}(${mode}+${grid})/specs_data.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --exclude=${EXCLUDE}

cd /sciclone/scr-lst/yacahuanamedra/GP/

source ~/.bashrc
module load miniforge3/24.9.2-0
conda init
sleep 5
conda activate gptorch

echo "Python path: \$(which python3)"
python3 --version

echo "Running on node: \$(hostname)"
echo "Running on CPUs: \$(scontrol show hostnames \$SLURM_NODELIST)"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "In directory: $(pwd)"


echo "Starting Python at $(date)"
python3 run_IS.py --mean "$model" --ker "$kernel" --mode "$mode" --grid "$grid" 
echo "Finished Python at $(date)"
EOF
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "Job $JOB_ID submitted successfully."
            echo "Running $model + $kernel in mode=$mode, grid=$grid"
            rm -f "$SLURM_SCRIPT"

            echo "SLURM script $SLURM_SCRIPT deleted."
        else
            echo "Failed to submit job. SLURM script not deleted: $SLURM_SCRIPT"
        fi

      done
    done
  done
done

