#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python singleSubjectTraining2.py $c > Logs\\Finetune\\shallow-cropped\\shallow-cropped-subject$c-400epoches.txt
done