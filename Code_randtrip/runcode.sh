#!/bin/bash
#PBS -N  fake-spec-rand
#PBS -l nodes=1:ppn=24
#PBS -o  output-file
#PBS -e  error-file
#PBS -m e 
#PBS -M raknath@gmail.com
#PBS -l walltime=12:00:00
echo "start time:"
date
export TERM=xterm
python $PBS_O_WORKDIR/run_train.py
echo "End time:"
date
clear
