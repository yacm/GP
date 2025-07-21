#!/bin/sh

#SBATCH --job-name=set_up
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --output=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/nnpdf.log
#SBATCH --error=/sciclone/scr-lst/yacahuanamedra/GP/ToDo/nnpdf.log

# Ensure the script is executed with exactly 3 arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <param2> <param3> <param4>"
    echo "Example: $0 Krbflog all log_lin"
    exit 1
fi

# Assigning arguments


PARAM1=$1  # Example: g_flat
PARAM2=$2  # Example: Krbflog
PARAM3=$3  # Example: all
PARAM4=$4  # Example: log_lin
PARAM5=$5  # Example: 20

# Submitting jobs for Re components
sbatch run1.csh $PARAM1 $PARAM2 Re $PARAM5 500000 $PARAM3 $PARAM4 14
#stopping 3 seconds
sleep 10
sbatch run1.csh $PARAM1 $PARAM2 Re $PARAM5 500000 $PARAM3 $PARAM4 13
#stopping 3 seconds
sleep 10
sbatch run1.csh $PARAM1 $PARAM2 Re $PARAM5 500000 $PARAM3 $PARAM4 12
sleep 10
# Submitting jobs for Im components
sbatch run1.csh $PARAM1 $PARAM2 Im $PARAM5 500000 $PARAM3 $PARAM4 12
sleep 10
sbatch run1.csh $PARAM1 $PARAM2 Im $PARAM5 500000 $PARAM3 $PARAM4 13
sleep 10
sbatch run1.csh $PARAM1 $PARAM2 Im $PARAM5 500000 $PARAM3 $PARAM4 14
sleep 10