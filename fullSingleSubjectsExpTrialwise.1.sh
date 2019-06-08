#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python ExperimentSingleSubjects.1.py $c > single_subjects\\deep-cropped-subject$c-2.5sec-800epoches.txt
done