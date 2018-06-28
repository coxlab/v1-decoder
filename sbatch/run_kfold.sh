#!/bin/bash

#SBATCH -p cox
#SBATCH --gres=gpu:1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=34000
#SBATCH -t 1000
#SBATCH -o /n/regal/cox_lab/dapello/slurm/kfold_%j.out
#SBATCH -e /n/regal/cox_lab/dapello/slurm/kfold_%j.err


cd /n/regal/cox_lab/dapello/v1-decoder
module load Anaconda3 
source activate my_root
KERAS_BACKEND=tensorflow

python -u Kfold_multi_session_decode.py /n/coxfs01/ephys/GRat31/v1_decoder_config_061118_kfold_light_config.json