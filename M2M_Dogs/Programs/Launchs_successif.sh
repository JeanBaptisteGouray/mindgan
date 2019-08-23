#!/bin/bash

if [ $# != '2' ]
then
    read -p "Combien de secondes voulez-vous attendre deux lancements ? " sec
    read -p "Quel script python voulez-vous lancer? " prog
    echo -- '\n'
    launch_local=1
else 
    sec=$1
    prog=$2
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

echo 'Memoire libre minimum :' $mem_min
echo 'Le PID est :' $$

if [ ! -e "../Logs_$prog" ]
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

    # ./notification_discord.sh "$i / $nb_wgan | Lancement d'un nouvel entrainement de $prog le $launch"

    sleep $sec
done

while [ -n "$PID" -a -e /proc/$PID ]
        do
            echo -ne '\rWait end of training!'
            sleep $sec
        done