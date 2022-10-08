#!/bin/bash

# TODO - run your inference Python3 code

#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"

#python3 train_1-1.py ${1} ${2} --mode="test" --model_option="A"
python3 train_1-1.py ${1} ${2} --mode="test" --model_option="B"

python3 train_1-1.py ./hw1_data/hw1_data/p1_data/train_50 . --mode="test" --model_option="B"



