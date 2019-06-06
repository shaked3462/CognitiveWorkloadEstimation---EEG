#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python ExperimentSingleSubjects.py $c > single_subjects\\shallow-trialwise-subject$c-2.5sec-800epoches.txt
done