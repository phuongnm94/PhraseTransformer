#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N ph-zhvi3
#PBS -j oe -l select=1
#PBS -q GPU-1
#PBS -o pbs_infer-sp.log
#PBS -e infer-sp.err.log
#PBS -M phuongnm@jaist.ac.jp 
#PBS -m e 

cd $PBS_O_WORKDIR

source ~/.bashrc
  
conda activate /home/phuongnm/syntaxNMT/env-nmt && \
cd /home/phuongnm/syntaxNMT/ && \
bash /home/phuongnm/syntaxNMT/data/zhvi/phrase-zhvi3/run_sota_phrasetrans.sh


wait
echo "All done"