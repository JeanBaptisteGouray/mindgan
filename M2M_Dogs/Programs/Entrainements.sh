#!/bin/bash

./Launchs_simultanes_alarm.sh 60 classifier_AE.py 1 0

./Launchs_simultanes_alarm.sh 60 classifier_MindGAN.py 1 0

./Launchs_simultanes_alarm.sh 60 train_AE.py 1 0

./Launchs_simultanes_alarm.sh 60 train_MindGAN.py 1 0