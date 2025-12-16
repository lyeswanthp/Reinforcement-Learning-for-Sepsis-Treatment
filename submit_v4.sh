#!/bin/bash
#SBATCH --job-name=run_double_dqn
#SBATCH --nodes=1
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gpus=1
#SBATCH --time=100:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=16        
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lovelyyeswanth2002@gmail.com

python run_double_dqn.py