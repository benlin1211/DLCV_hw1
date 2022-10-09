#!/bin/bash

# TODO - run train_1-1.py

# B-1
#python3 train_1-2.py ${1} ${2} --mode="train" --model_option="B" --learning_rate=2e-4 --l2_reg_lambda=5e-5 --weight_decay=1e-8 --checkpth="ckpt_seg2Bsh" 

# B-2
python3 train_1-2.py ${1} ${2} --mode="train" --model_option="B" --learning_rate=2e-4 --l2_reg_lambda=5e-5 --weight_decay=1e-8 --checkpth="ckpt_seg2Bsh_no_weight"
