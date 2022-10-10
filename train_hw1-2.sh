#!/bin/bash

# TODO - run train_1-1.py

# A
# python3 train_1-2.py ${1} ${2} --mode="train" --model_option="A" --l2_reg_lambda=0.0 --checkpth="ckpt_1-2A"  --n_split=10

# B-1
# python3 train_1-2.py ${1} ${2} --mode="train" --model_option="B" --learning_rate=2e-4 --l2_reg_lambda=0.0 --checkpth="ckpt_1-2B-1"  --n_split=10

# B-2
# 記得 x3 x4 改成沒有 weight
# python3 train_1-2.py ${1} ${2} --mode="train" --model_option="B" --learning_rate=2e-4 --l2_reg_lambda=0.0 --checkpth="ckpt_1-2B-2"  --n_split=10

# C
# best record: [0.68] python3 train_1-2.py ${1} ${2} --mode="train" --model_option="C" --learning_rate=5e-4 --l2_reg_lambda=0.0 --checkpth="ckpt_1-2C"  --n_split=10
# 記得 x3 x4 改成沒有 weight
#python3 train_1-2.py ${1} ${2} --mode="train" --model_option="C" --learning_rate=5e-4 --scheduler_lr_decay_ratio=0.9 --checkpth="ckpt_1-2-deeplab"  --n_split=10
python3 train_1-2.py ${1} ${2} --mode="train" --model_option="C" --learning_rate=2e-4 --l2_reg_lambda=1e-5 --scheduler_lr_decay_ratio=0.95 --checkpth="ckpt_1-2-deeplab"  --n_split=10

#python3 train_1-2.py ${1} ${2} --mode="train" --model_option="C" --learning_rate=5e-4 --scheduler_lr_decay_ratio=0.9 --checkpth="ckpt_1-2-deeplab_new2"  --n_split=10

