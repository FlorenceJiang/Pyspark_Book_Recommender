module load anaconda3
conda create -n venv python=3.6
source activate /home/dj1369/.conda/envs/venv
conda install -c conda-forge python-annoy

export PYSPARK_PYTHON=/home/dj1369/.conda/envs/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/dj1369/.conda/envs/venv/bin/python

## This is the commandline to create virtual env and install annoy on Dumbo
## The last two lines are used to set pyspark python to the one in venv, in which annoy is installed using conda.