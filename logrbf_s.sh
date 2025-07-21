#!/bin/bash

# Loop to explore values from 5.0 to 50.0 in steps of .5
for value in $(seq 20 1 80)
do
    # Build the parameter string with the current value
    parameter="Krbflog_no_sn=${value}.0"

    # Run the sbatch command
    sbatch run_IS.csh "PDF_N g_flat" "all" "log_lin" "$parameter"

    echo "Submitted: sbatch run_IS.csh \"PDF_N g_flat\" \"all\" \"log_lin\" \"$parameter\""
done