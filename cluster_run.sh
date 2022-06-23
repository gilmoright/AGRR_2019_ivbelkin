#!/bin/bash

#SBATCH -J distilconvv1
#SBATCH -D /s/ls4/users/grartem/Gapping/AGRR_2019_ivbelkin
#SBATCH -o /s/ls4/users/grartem/Gapping/AGRR_2019_ivbelkin/logs/%x_%j.out
#SBATCH -e /s/ls4/users/grartem/Gapping/AGRR_2019_ivbelkin/logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 3
#SBATCH --gres=gpu:k80:1
#SBATCH --time=72:00:00

export HOME=/s/ls4/users/grartem
#export PATH=$HOME/anaconda3/envs/simptr/bin:$PATH
export PATH=$HOME/anaconda3/bin:$PATH
module load gcc/7.3.0
export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:/usr/lib64/nvidia:/usr/lib64:
	   #LD_LIBRARY_PATH=/s/ls4/sw/cuda-9.0/lib64:/s/ls4/sw/cuda-9.0/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
./run.sh