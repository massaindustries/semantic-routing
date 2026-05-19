#!/usr/bin/env bash

# Loop over each of the three directories
for folder in "verified" "test"
do
  # Find all subdirectories (the trailing slash ensures only directories are returned)
  for MODEL_DIR in "evaluation/$folder/"*/
  do
    # Double-check that it is indeed a directory and that model_dir contains the word "sweagent"
    if [[ -d "$MODEL_DIR" && "$MODEL_DIR" == *sweagent* ]]; then
      MODEL_NAME=$(basename "$MODEL_DIR")
      echo "Processing logs for model: $MODEL_NAME (folder: $folder)"

      # Run the command
      python -m analysis.download_logs "evaluation/$folder/$MODEL_NAME" --only_trajs --skip_existing --use_cli
      
      # Check if the command was successful
      if [ $? -ne 0 ]; then
        echo "Error occurred while processing $MODEL_NAME in $folder."
      fi
    fi
  done
done