#!/bin/bash

# Initialize variables
MODEL_NAME="your_path_to/Qwen-VL-Chat"
DATA_DIR=""

# Process command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)  # unknown option
            echo "Unknown option: $1"
            echo "Usage: $0 [--model_name MODEL_NAME] [--data_dir DATA_DIR]"
            exit 1
            ;;
    esac
done

# Construct the command line
cmd="python experiments/main.py"
if [ ! -z "$MODEL_NAME" ]; then
    cmd+=" --model_name $MODEL_NAME"
fi
if [ ! -z "$DATA_DIR" ]; then
    cmd+=" --data_dir $DATA_DIR"
fi

# Navigate to the project root; adjust this if your script is somewhere else
cd "$(dirname "$0")/../../" || exit
export PYTHONPATH=$(pwd)

# Run the Python script with the constructed command line
echo "Running command: $cmd"
$cmd
