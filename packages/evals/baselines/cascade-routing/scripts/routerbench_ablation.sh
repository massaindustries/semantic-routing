#!/bin/bash

# Define the step size and maximum value
STEP=0.2
MAX=3.0
MAX_JOBS=4

# Function to format numbers to one decimal place
format_num() {
    printf "%.1f" "$1"
}

# Export the function for use in subshells (if needed)
export -f format_num

# Iterate over x from 0 to MAX in steps of STEP
for x in $(seq 0 "$STEP" "$MAX"); do
    x_formatted=$(format_num "$x")
    
    # Iterate over y from 0 to x in steps of STEP
    for y in $(seq 0 "$STEP" "$x"); do
        y_formatted=$(format_num "$y")
        
        # Run the routerbench.py script in the background
        python scripts/routerbench.py --models 0,9,4,3,5 --noise-before "$x_formatted" --noise-after "$y_formatted" --extensive-file &
        
        # Check the number of active background jobs
        while true; do
            # Get the number of running background jobs
            RUNNING_JOBS=$(jobs -rp | wc -l)
            
            # If the number of running jobs is less than MAX_JOBS, break the loop
            if [ "$RUNNING_JOBS" -lt "$MAX_JOBS" ]; then
                break
            fi
            
            # Otherwise, wait for a short period before checking again
            sleep 1
        done
    done
done

# Wait for all background jobs to finish before exiting the script
wait

echo "All routerbench scripts have completed."