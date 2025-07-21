#!/bin/sh

#SBATCH --job-name=set_up
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/output.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/output.log


if [ "$#" -ne 7 ]; then
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

# Conditional check for ITD
if [ "$ITD" = "Im" ]; then
    data=15
else
    data=15
fi

if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm04,fm08,fm24"
else
    EXCLUDE=""
fi

CPU=$((TIMES * data - 1))
ITER=$((ITERATIONS / TIMES))
BURN=$((ITER/10))
#NEWSLURMID=$SLURM

# Create the folder if it doesn't exist
mkdir -p "${MODEL}_${KERNEL_NAME}(${mode}+${grid})"

# Create the SLURM script dynamicallyy
SLURM_SCRIPT="${MODEL}_${KERNEL_NAME}(${mode}+${grid})/job_script.slurm"

cat << EOF > $SLURM_SCRIPT
#!/bin/tcsh

#SBATCH --job-name=${MODEL}_${KERNEL_NAME}(${mode}+${grid})${ITD}
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/${MODEL}_${KERNEL_NAME}(${mode}+${grid})/specs/GP${ITD}_%a.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/${MODEL}_${KERNEL_NAME}(${mode}+${grid})/specs/GP${ITD}_%a.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${CPU}
#SBATCH --mem=1000M
#SBATCH --exclude=${EXCLUDE}

module load miniforge3/24.9.2-0
conda activate gptorch

@ NEWSLURMID = \$SLURM_ARRAY_TASK_ID % ${data}

echo "Running on \$NEWSLURMID"


python3 run.py --i \$NEWSLURMID --Nsamples ${ITER} --burn ${BURN} --L 100 --eps 0.001 --ITD ${ITD} --mean ${MODEL} --ker ${KERNEL_NAME} --mode ${mode}  --IDslurm \$SLURM_ARRAY_TASK_ID --grid ${grid} --Nx 256
EOF

# Submit the job
JOB_ID=$(sbatch $SLURM_SCRIPT | awk '{print $4}')

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Job $JOB_ID submitted successfully."
    echo " You are runing $MODEL + $KERNEL_NAME ($mode) with $ITD(M) in $TIMES chains of $ITER"
    # Remove the SLURM script after submission
    rm -f $SLURM_SCRIPT
    echo "SLURM script $SLURM_SCRIPT deleted."
else
    echo "Failed to submit job. SLURM script not deleted."
fi
