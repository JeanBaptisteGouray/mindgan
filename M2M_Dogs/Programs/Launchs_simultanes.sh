#!/bin/bash

nb_diviseur=4

if [ $# != '3' -a $# != '4' -a $# != '5' ]
then
    read -p "Combien de VanillaGAN voulez_vous entrainer simultannement ? " nb_wgan
    read -p "Combien de secondes voulez-vous attendre deux lancements ? " sec
    read -p "Quel script python voulez-vous lancer? " prog
    read -p "L'entrainement se fait sur GPU? [0/1]" train_on_gpu
    echo -- '\n'
    launch_local=1
else 
    nb_wgan=$1
    sec=$2
    prog=$3
    if [ $4 = '' ]
    then
        train_on_gpu=1
    else
        train_on_gpu=$4
    fi
    launch_local=$5
fi

if [ $train_on_gpu -ne 0 ]
then
    info=$(nvidia-smi --query-gpu=memory.total --format=csv) 
    List_info=(${info////})    
    free_mem=${List_info[2]}
    mem_min=$(($free_mem/$nb_diviseur))
else
    info=$(free -m | grep "Mem") 
    List_info=(${info////})    
    free_mem=${List_info[3]}
    mem_min=$(($free_mem/$nb_diviseur))
fi

echo 'Memoire libre minimum :' $mem_min
echo 'Le PID est :' $$

for ((i = 1; i <= $nb_wgan; i++))
do
    wait=0
    if [ $train_on_gpu -ne 0 ]
    then
        info=$(nvidia-smi --query-gpu=memory.free --format=csv) 
        List_info=(${info////})    
        free_mem=${List_info[2]}
    else
        info=$(free -m | grep "Mem") 
        List_info=(${info////})    
        free_mem=${List_info[3]}
    fi

    while [ $free_mem -le $mem_min ]
    do
        echo -ne '\rLOW MEMORY WAIT!'
        sleep $sec
        if [ $train_on_gpu -ne 0 ]
        then
            info=$(nvidia-smi --query-gpu=memory.free --format=csv) 
            List_info=(${info////})    
            free_mem=${List_info[2]}
        else
            info=$(free -m | grep "Mem") 
            List_info=(${info////})    
            free_mem=${List_info[3]}
        fi
        wait=1
    done

    if [ $launch_local -eq 1 ]
    then
        gnome-terminal --geometry=190x25 -- bash -c "python '$prog'"
    else
        dbus-launch gnome-terminal --geometry=190x25 -- bash -c "python '$prog'"
    fi

    launch=$(date)

    if [ $wait -eq 1 ]
    then
        echo -e '\n\n'
    fi
    echo -e $i '/' $nb_wgan 'training launched ' $launch '\t|\tFree memory : '$free_mem
    sleep $sec
done