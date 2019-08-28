#!/bin/bash

sec=$1
webhook=$2
folder_log=../Logs_successifs

python3 nb_lancement.py

if [ ! -d "$folder_log" ]
then
    mkdir $folder_log
fi

echo 'PID :' $$

nohup ./Launchs_successif.sh 30 classifier_AE.py $webhook &> $folder_log/log_succ_C_AE.log &

PID=$!
echo 'PID du Launchs_successif :' $PID

while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 classifier_MindGAN.py $webhook &> $folder_log/log_succ_C_M.log &

PID=$!
echo 'PID du Launchs_successif :' $PID

while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 train_AE.py $webhook &> $folder_log/log_succ_t_AE.log &

PID=$!
echo 'PID du Launchs_successif :' $PID

while [ -n "$PID" -a -e /proc/$PID ]
do
    echo -ne '\rWait end of training!'
    sleep $sec
    wait=1
done

nohup ./Launchs_successif.sh 30 train_MindGAN.py $webhook &> $folder_loglog_succ_t_M.log &