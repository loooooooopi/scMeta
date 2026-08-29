#!/bin/bash
#SBATCH -J make_figure2
#SBATCH -A pccr
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 4:00:00
#SBATCH -o /depot/natallah/data/Mengbo/scMetas/revision3/Github/figure_regen/make_figure2.log
#SBATCH -e /depot/natallah/data/Mengbo/scMetas/revision3/Github/figure_regen/make_figure2.log
cd /depot/natallah/data/Mengbo/scMetas/revision3/Github/figure_regen/
/depot/natallah/data/Mengbo/scMetas/revision2/scmeta_env/bin/python3 -u make_figure2.py
