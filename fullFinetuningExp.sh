#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python ExperimentNirDataset.py $c > finetuning\\shallow-trialwise-subject$c-2.5sec-800epoches.txt
done
