#!/bin/bash

# TODO - run train_1-1.py

#python3 train_1-1.py ${1} ${2} --mode="train" --model_option="B" --l2_reg_lambda=0.005
python3 train_1-1.py ${1} ${2} --mode="train" --model_option="A" --l2_reg_lambda=0.005 --checkpth="ckpt"



#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done
