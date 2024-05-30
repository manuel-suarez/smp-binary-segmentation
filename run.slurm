#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-v5
#SBATCH --time=0
#SBATCH --mem=0

/home/est_posgrado_manuel.suarez/anaconda3/envs/py38-sm-pt/bin/python main.py
