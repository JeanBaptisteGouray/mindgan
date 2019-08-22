#!/bin/bash

nohup ./Launchs_successif.sh 30 classifier_AE.py &> log_succ_C_AE.log &

nohup ./Launchs_successif.sh 30 classifier_MindGAN.py &> log_succ_C_M.log &

nohup ./Launchs_successif.sh 30 train_AE.py &> log_succ_t_AE.log &

nohup ./Launchs_successif.sh 30 train_MindGAN.py  &> log_succ_t_M.log &