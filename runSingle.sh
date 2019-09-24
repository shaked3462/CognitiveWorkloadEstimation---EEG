#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python ExperimentSingleSubjects.py $c > Logs\\SingleSubject\\shallow-cropped-subject$c-400epoches.txt
done