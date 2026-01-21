#!/bin/bash
'''
    Please do not remove this file as it is required for tdspds initiation.
'''
source /venv/bin/activate

# Run the main script with the parsed parameters
export PYTHONPATH="${PYTHONPATH}:/opt/ml/code/"
echo $PYTHONPATH

echo "Hello you have provided the following arguments: " "$@"
/opt/ml/code/src/initialize_tdspds.sh

# Determine which training script to run based on MODEL_TYPE environment variable
# MODEL_TYPE can be: computer_vision (cv) or tabular (default)
MODEL_TYPE="${MODEL_TYPE:-computer_vision}"

echo "Model Type: $MODEL_TYPE"

if [ "$MODEL_TYPE" = "computer_vision" ] || [ "$MODEL_TYPE" = "cv" ]; then
    echo "Running Computer Vision Model Training..."
    python /opt/ml/code/src/train_cv.py $*
else
    echo "Running Tabular Model Training (Churn Prediction)..."
    python /opt/ml/code/src/train.py $*
fi
#===============================================================================================================
