#!/bin/bash
for (( c = 1; c <= 52; c++ ))
do  
   python fintuneShallowCropped.py $c > finetuning\\shallow-cropped-subject$c-2.5sec-800epoches.txt
done
