#!/bin/bash

# TODO - run your inference Python3 code

#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done
# download model
bash ./hw1_download.sh 

python3 train_1-1.py ${1} ${2} --mode="test" --model_option="A"
python3 train_1-1.py ${1} ${2} --mode="test" --model_option="B"
