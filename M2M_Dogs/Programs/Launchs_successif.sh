#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

if [ $# != '2' -a $# != '3' ]
then
    read -p "Combien de secondes voulez-vous attendre deux lancements ? " sec
    read -p "Quel script python voulez-vous lancer? " prog
    echo -- '\n'
else 
    sec=$1
    prog=$2
fi

if [ ! -z $3 ]
then 
    case $3 in
        "LENS" | "Lens" | "lens") num_ligne=0
            ;;
        "Nvidia" | "NVIDIA" | "nvidia") num_ligne=1
            ;;
    esac
else 
    num_ligne=0 
fi

fichier=${prog%.*}
fichier=${fichier#t*_}

nb_wgan=`cat ../../Nb_lancement/${fichier%_*}.txt`

if [ -d "../${fichier^}/Trainings" ]
then
    folders=$(ls ../${fichier^}/Trainings)
    folders=(${folders////})
    nb_wgan_test=${#folders[@]}
else
    nb_wgan_test=0
fi

nb_wgan=$(($nb_wgan - $nb_wgan_test))

webhooks=`cat webhooks.txt`

webhooks_possible=()

for webhook in $webhooks
do
    webhooks_possible=( "${webhooks_possible[@]}" "$webhook" )
done

webhook=${webhooks_possible[num_ligne]#*:}

unset webhooks webhooks_possible

echo 'Le PID est :' $$

if [ ! -d "../Logs_$prog" ]
then
    mkdir ../Logs_$prog
fi

for ((i = 1; i <= $nb_wgan; i++))
do
    wait=0
    if [ $i -ne 1 ]
    then
        while [ -n "$PID" -a -e /proc/$PID ]
        do
            echo -ne '\rWait end of training!'
            sleep $sec
            wait=1
        done
    fi

    nohup python3 $prog &> ../Logs_$prog/log_$i.log &

    PID=$!
    launch=$(date)

    if [ $wait -eq 1 ]
    then
        echo -e '\n\n'
    fi
    echo -e $i '/' $nb_wgan 'training launched ' $launch ' | PID:' $PID 

    ./notification_discord.sh "$i / $nb_wgan | Lancement d'un nouvel entrainement de $prog le $launch" "$webhook"

    sleep $sec
done

while [ -n "$PID" -a -e /proc/$PID ]
        do
            echo -ne '\rWait end of training!'
            sleep $sec
done