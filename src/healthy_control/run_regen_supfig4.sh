#!/bin/bash
#SBATCH -J regen_supfig4
#SBATCH -A pccr
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -t 12:00:00
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/regen_supfig4.log
#SBATCH -e /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/regen_supfig4.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/src/healthy_control/
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 -u regen_supfig4.py
