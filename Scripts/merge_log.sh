##!/bin/bash
root=$1

programs=( "Classifier_AE" "Classifier_MindGAN" "AE" "MindGAN" )

cd $root

if [ ! -d "$root/Results" ]; then
    mkdir $root/Results
fi

for program in ${programs[@]}
do
    if [ -d $program ]
    then
        if [ ! -d "$root/Results/$program" ]; then
            mkdir $root/Results/$program
            mkdir $root/Results/$program/Logs
        fi
    fi
done

for program in ${programs[@]}
do
    if [ -d $program ]
    then
        folders=$(ls $program/Trainings)
        for folder in $folders
        do
            cp $program/Trainings/$folder/log.csv Results/$program/Logs/$folder.csv
        done
    fi
done