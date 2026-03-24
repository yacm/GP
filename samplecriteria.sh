#!/bin/bash

# Usage check
if [ $# -ne 6 ]; then
    echo "Usage: $0 <kernelname> <initial> <final> <step>"
    echo "Example: $0 rbf_logrbf_ln= 1.0 3.0 0.5"
    exit 1
fi

# Parameters
kernelname=$1   # e.g., rbf_logrbf_ln=
initial=$2      # e.g., 1.0
final=$3        # e.g., 3.0
step=$4         # e.g., 0.5
data=$5
modes=$6

# Loop over floating-point values using awk
value=$initial
while (( $(echo "$value <= $final" | bc -l) )); do
    # Format value to 2 decimal
    value_formatted=$(printf "%.1f" "$value")
    parameter="${kernelname}${value_formatted}"

    # Submit job #linh is the new grid
    sbatch run_IS.csh "PDF_N g_flat" "$modes" "log_lin" "$parameter" "$data"
    echo "Submitted: sbatch run_IS.csh \"PDF_N g_flat\" \"$modes\" \"log_lin\" \"$parameter\" \"$data\""

    # Increment
    value=$(echo "$value + $step" | bc -l)
done