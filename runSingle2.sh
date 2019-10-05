#!/bin/bash
for (( c = 14; c <= 52; c++ ))
do  
   python singleSubjectTraining2.py $c > Logs\\SingleSubject\\deep-cropped\\deep-cropped-subject$c-400epoches.txt
done