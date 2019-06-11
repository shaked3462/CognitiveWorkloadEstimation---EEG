#!/bin/bash

python fintuneDeepCropped.py 42 > finetuning\\deep-cropped-subject42-2.5sec-800epoches.txt
python fintuneDeepCropped.py 44 > finetuning\\deep-cropped-subject44-2.5sec-800epoches.txt
python fintuneDeepCropped.py 45 > finetuning\\deep-cropped-subject45-2.5sec-800epoches.txt

for (( c = 48; c <= 52; c++ ))
do  
   python fintuneDeepCropped.py $c > finetuning\\deep-cropped-subject$c-2.5sec-800epoches.txt
done
