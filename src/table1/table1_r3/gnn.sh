#!/bin/bash
#SBATCH -p a10
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH -J t1_gnn
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/logs/gnn_%j.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 table1_r3_gnn.py --shard ${SHARD:-0} --nshards ${NSHARDS:-3}
