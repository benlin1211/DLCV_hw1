#!/bin/bash

# TODO - run your inference Python3 code

#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done

# best(09/26 14:14): 
#python3 train_1-1.py ${1} --model_option="B" --l2_reg_lambda=0.005

# for resnet50
python3 train_1-1.py ${1} --model_option="B" --l2_reg_lambda=1e-5
