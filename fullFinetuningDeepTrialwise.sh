#!/bin/bash
for (( c = 50; c <= 52; c++ ))
do  
   python fintuneDeepTrialwise.py $c > finetuning\\deep-trialwise-subject$c-2.5sec-800epoches.txt
done
