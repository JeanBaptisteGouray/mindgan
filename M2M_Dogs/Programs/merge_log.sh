##!/bin/bash
root=/Users/yaelfregier/Git/WIP/M2M_Dogs

programs=( "Classifier_AE" "Classifier_MindGAN" )

cd $root

if [ ! -d "$root/Results" ]; then
    mkdir $root/Results
fi

for program in ${programs[@]}
do
    if [ ! -d "$root/Results/$program" ]; then
        mkdir $root/Results/$program
        mkdir $root/Results/$program/Logs
    fi
done

for program in ${programs[@]}
do
    folders=$(ls $program/Trainings)
    for folder in $folders
    do
        cp $program/Trainings/$folder/log.csv Results/$program/Logs/$folder.csv
    done
done