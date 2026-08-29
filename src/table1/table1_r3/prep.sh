#!/bin/bash
#SBATCH -A pccr
#SBATCH -p a10
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/logs/prep.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 table1_r3_prep.py
