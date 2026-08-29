#!/bin/bash
#SBATCH -J healthy_v3
#SBATCH -A pccr
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -t 12:00:00
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/test_healthy_v3.log
#SBATCH -e /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/test_healthy_v3.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 -u test_healthy_v3.py
