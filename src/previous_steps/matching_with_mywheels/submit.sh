#!/bin/bash
#SBATCH -n 56 
#SBATCH -t 720:00:00
#SBATCH -J match 
#SBATCH -o match.slurm
#SBATCH -A phy200025p
#SBATCH -p HENON
# use the bash shell
set -x 
time python match.py

