#!/bin/bash
for (( c=7; c<=52; c++ ))
do  
   python ExperimentSingleSubjects.py $c > single_subjects\\shallow-trialwise-subject$c-2.5sec.txt
done