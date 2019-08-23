#!/bin/bash

sec=$1

nohup ./Launchs_successif.sh 30 classifier_AE.py &> log_succ_C_AE.log &

PID=$!
while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 classifier_MindGAN.py &> log_succ_C_M.log &

PID=$!
while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 train_AE.py &> log_succ_t_AE.log &

PID=$!
while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 train_MindGAN.py  &> log_succ_t_M.log &