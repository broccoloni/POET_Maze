#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-aali
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=32G
source ~/lgraha/bin/activate
cd ~/scratch/POET_Maze/
python runpoet.py 0 0 1 &>> poetv0eps0reps1.out
