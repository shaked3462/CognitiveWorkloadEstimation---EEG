#!/bin/bash
<<<<<<< HEAD
for (( c = 45; c <= 52; c++ ))
=======
for (( c = 37; c <= 52; c++ ))
>>>>>>> 918e554f2a3984a267b664cc2e217b23dccc8149
do  
   python fintuneDeepTrialwise.py $c > finetuning\\deep-trialwise-subject$c-2.5sec-800epoches.txt
done
