#!/bin/bash
for (( c = 51; c <= 52; c++ ))
do  
   python fintuneDeepCropped.py $c > finetuneCrossSubjects\\deep-cropped-subject$c-2.5sec-800epoches.txt
done
