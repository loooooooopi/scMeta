#!/bin/bash
#SBATCH -A ayg
#SBATCH -p a10
#SBATCH -t 24:00:00
#SBATCH -c 24
#SBATCH --gres=gpu:1
#SBATCH --mem=350G
#SBATCH -J t1_classical
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/logs/classical.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1
export OMP_NUM_THREADS=2
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 table1_r3_classical.py --workers 6
