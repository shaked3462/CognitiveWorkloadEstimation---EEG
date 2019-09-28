#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python singleSubjectTraining.py $c > Logs\\Finetune\\shallow-trialwise\\shallow-trialwise-subject$c-400epoches.txt
done