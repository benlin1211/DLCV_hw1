#!/bin/bash

# TODO - run train_1-1.py
#     parser.add_argument("src", help="Training data location")
#     parser.add_argument("dest", help="CSV prediction output location (for test mode)")


# best(09/26 14:14): 
#python3 train_1-1.py ${1} ${2} --mode="train" --model_option="B" --l2_reg_lambda=0.005

# 
# python3 train_1-1.py ${1} ${2} --mode="train" --model_option="A" --l2_reg_lambda=0.005

python3 train_1-1.py ${1} ${2} --mode="test" --model_option="A"
python3 train_1-1.py ${1} ${2} --mode="test" --model_option="B"


#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done