#!/bin/bash

# Usage check
if [ $# -ne 4 ]; then
    echo "Usage: $0 <kernelname> <initial> <final> <step>"
    echo "Example: $0 rbf_logrbf_ln= 1.0 3.0 0.5"
    exit 1
fi

# Parameters
kernelname=$1   # e.g., rbf_logrbf_ln=
initial=$2      # e.g., 1.0
final=$3        # e.g., 3.0
step=$4         # e.g., 0.5

# Loop over floating-point values using awk
value=$initial
while (( $(echo "$value <= $final" | bc -l) )); do
    # Format value to 2 decimal
    value_formatted=$(printf "%.1f" "$value")
    parameter="${kernelname}${value_formatted}"

    # Submit job
    sbatch run_IS.csh "PDF_N g_flat" "all kernel mean" "log_lin" "$parameter"
    echo "Submitted: sbatch run_IS.csh \"PDF_N g_flat\" \"all kernel mean\" \"log_lin\" \"$parameter\""

    # Increment
    value=$(echo "$value + $step" | bc -l)
done