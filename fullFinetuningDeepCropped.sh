#!/bin/bash
for (( c = 46; c <= 52; c++ ))
do  
   python fintuneDeepCropped.py $c > finetuning\\deep-cropped-subject$c-2.5sec-800epoches.txt
done