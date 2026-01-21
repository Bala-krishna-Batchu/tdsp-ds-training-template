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

python /opt/ml/code/src/train.py $*
#===============================================================================================================
