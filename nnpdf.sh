#!/bin/sh
# Ensure the script is executed with exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <param1> <param2> <param3>"
    echo "Example: $0 Krbflog all log_lin"
    exit 1
fi

# Assigning arguments
PARAM1=$1  # Example: Krbflog
PARAM2=$2  # Example: all
PARAM3=$3  # Example: log_lin

# Submitting jobs for Re components
sbatch run1.csh PDF $PARAM1 Re 20 80000 $PARAM2 $PARAM3 14
#stopping 3 seconds
sleep 5
sbatch run1.csh PDF $PARAM1 Re 20 80000 $PARAM2 $PARAM3 13
#stopping 3 seconds
sleep 5
sbatch run1.csh PDF $PARAM1 Re 20 80000 $PARAM2 $PARAM3 12
sleep 5
# Submitting jobs for Im components
sbatch run1.csh PDF $PARAM1 Im 20 80000 $PARAM2 $PARAM3 12
sleep 5
sbatch run1.csh PDF $PARAM1 Im 20 80000 $PARAM2 $PARAM3 13
sleep 5
sbatch run1.csh PDF $PARAM1 Im 20 80000 $PARAM2 $PARAM3 14
sleep 5