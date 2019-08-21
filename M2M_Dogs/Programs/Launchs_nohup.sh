#!/bin/bash

# nb_diviseur=3

if [ $# != '2' -a $# != '3'  ]
then
    read -p "Combien de secondes voulez-vous attendre deux lancements ? " sec
    read -p "Quel script python voulez-vous lancer? " prog
    read -p "L'entrainement se fait sur GPU? [0/1]" train_on_gpu
    echo -- '\n'
    launch_local=1
else 
    sec=$1
    prog=$2
    if [ $3 = '' ]
    then
        train_on_gpu=1
    else
        train_on_gpu=$3
    fi
fi

fichier=${prog%.*}

nb_wgan=`cat ../../Nb_lancement/$fichier.txt`

fichier=${fichier#t*_}

if [ -d "../${fichier^}/Trainings" ]
then
    folders=$(ls ../${fichier^}/Trainings)
    folders=(${folders////})
    nb_wgan_test=${#folders[@]}
else
    nb_wgan_test=0
fi

nb_wgan=$(($nb_wgan - $nb_wgan_test))

if [ $train_on_gpu -ne 0 ]
then
    # info=$(nvidia-smi --query-gpu=memory.total --format=csv) 
    # List_info=(${info////})    
    # free_mem=${List_info[2]}
    # mem_min=$(($free_mem/$nb_diviseur))
    mem_min=3000
else
    # info=$(free -m | grep "Mem") 
    # List_info=(${info////})    
    # free_mem=${List_info[3]}
    # mem_min=$(($free_mem/$nb_diviseur))
    mem_min=3000
fi

echo 'Memoire libre minimum :' $mem_min
echo 'Le PID est :' $$

if [ ! -e "../Logs_$prog" ]
then
    mkdir ../Logs_$prog
fi

for ((i = 1; i <= $nb_wgan; i++))
do
    wait=0
    if [ $train_on_gpu -ne 0 ]
    then
        info=$(nvidia-smi --query-gpu=memory.free --format=csv) 
        info=(${info////})
        free_mem=0
        k=2
        while [ $k -lt ${#info[@]} ]
        do
            free_mem=$((${info[$k]}))
            if [ $free_mem -ge $mem_min ]
            then
                GPU=$((k/2-1))
                k=${#info[@]}
            else
                k=$((k+2))
            fi
        done
    else
        info=$(free -m | grep "Mem") 
        info=(${info////})    
        free_mem=${info[3]}
    fi

    while [ $free_mem -le $mem_min ]
    do
        echo -ne '\rLOW MEMORY WAIT!'
        sleep $sec
        if [ $train_on_gpu -ne 0 ]
        then
            info=$(nvidia-smi --query-gpu=memory.free --format=csv) 
            info=(${info////})
            free_mem=0
            k=2
            while [ $k -lt ${#info[@]} ]
            do
                free_mem=$((${info[$k]}))
                if [ $free_mem -ge $mem_min ]
                then
                    GPU=$((k/2-1))
                    k=${#info[@]}
                else
                    k=$((k+2))
                fi
            done
        else
            info=$(free -m | grep "Mem") 
            List_info=(${info////})    
            free_mem=${List_info[3]}
        fi
        wait=1
    done

    CUDA_VISIBLE_DEVICES="$GPU" nohup python3 $prog &> ../Logs_$prog/log_$i.log &
    
    launch=$(date)

    if [ $wait -eq 1 ]
    then
        echo -e '\n\n'
    fi
    echo -e $i '/' $nb_wgan 'training launched ' $launch ' | PID:' $! ' | GPU: ' $GPU ' | Free memory: '$free_mem

    ./notification_discord.sh "$i / $nb_wgan | Lancement d'un nouvel entrainement de $prog le $launch"

    sleep $sec
done