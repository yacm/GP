#!/bin/sh

#SBATCH --job-name=set_up
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/run1.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/run1.log


if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <KERNEL_NAME>"
    exit 1
fi

MODEL=$1
KERNEL_NAME=$2
ITD=$3
TIMES=$4
ITERATIONS=$5
mode=$6
grid=$7
II=$8

# Conditional check for ITD
if [ "$ITD" = "Im" ]; then
    data=15
else
    data=15
fi

CPU=$((TIMES - 1)) #How many CPUs are you going to use
ITER=$((ITERATIONS / TIMES)) #How many iterations per cpu are you going to produce
#NEWSLURMID=$SLURM
BURN=$((ITER/10))
AA=$((II + 1))
# if hostname is fm then exclude fm04,fm08
if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm04,fm08,fm24"
else
    EXCLUDE=""
fi

# Create the folder if it doesn't exist
mkdir -p "${MODEL}_${KERNEL_NAME}(${mode}+${grid})"

# Create the SLURM script dynamically
SLURM_SCRIPT="${MODEL}_${KERNEL_NAME}(${mode}+${grid})/job_script.slurm"

cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=${MODEL}_${KERNEL_NAME}(${mode}+${grid})${ITD}(z=${AA}a)
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/${MODEL}_${KERNEL_NAME}(${mode}+${grid})/specs1/GP${ITD}(z=${AA}a)_%a.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/${MODEL}_${KERNEL_NAME}(${mode}+${grid})/specs1/GP${ITD}(z=${AA}a)_%a.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${CPU}
#SBATCH --mem=1000M
#SBATCH --exclude=${EXCLUDE}

source ~/.bashrc
module load miniforge3/24.9.2-0
conda init
sleep 5
conda activate gptorch

echo "Running on node: \$(hostname)"
echo "Running on CPUs: \$(scontrol show hostnames \$SLURM_NODELIST)"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "SLURM array job ID: \$SLURM_ARRAY_JOB_ID"
echo "SLURM array task ID: \$SLURM_ARRAY_TASK_ID"

python3 run.py --i ${II} --Nsamples ${ITER} --burn ${BURN} --L 600 --eps 0.001 --ITD ${ITD} --mean ${MODEL} --ker ${KERNEL_NAME} --mode ${mode}  --IDslurm \$SLURM_ARRAY_TASK_ID --grid ${grid} --Nx 256
EOF

# Submit the job
JOB_ID=$(sbatch $SLURM_SCRIPT | awk '{print $4}')

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Job $JOB_ID submitted successfully."
    echo " You are runing $MODEL + $KERNEL_NAME with $ITD(M) in $TIMES chains of $ITER with z=${AA}a"
    # Remove the SLURM script after submission
    rm -f $SLURM_SCRIPT
    echo "SLURM script $SLURM_SCRIPT deleted."
else
    echo "Failed to submit job. SLURM script not deleted."
fi