#!/bin/bash
#SBATCH --job-name=simplified_rl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # Request 16 CPU cores for parallel processing
#SBATCH --mem=128G                  # Request 128GB RAM (MIMIC-IV needs lots of memory)
#SBATCH --time=47:00:00             # Allow up to 24 hours
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lovelyyeswanth2002@gmail.com

python main.py
