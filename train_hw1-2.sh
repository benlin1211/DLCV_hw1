#!/bin/bash

# TODO - run train_1-1.py

python3 train_1-2.py ${1} ${2} --mode="train" --model_option="A" --l2_reg_lambda=0 --n_split=10

