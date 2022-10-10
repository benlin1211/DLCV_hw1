#!/bin/bash

# TODO - run your inference Python3 code
# Test: generate image one by one.

# A
# python3 train_1-2.py ${1} ${2} --mode="test" --model_option="A" --batch_size=1  --checkpth="ckpt_seg"
# B-1
#python3 train_1-2.py ${1} ${2} --mode="test" --model_option="B" --batch_size=1  --checkpth="ckpt_seg2Bsh" 

# B-2
# 記得 x3 x4 改成沒有 weight
# python3 train_1-2.py ${1} ${2} --mode="test" --model_option="B" --batch_size=1  --checkpth="ckpt_seg2Bsh_no_weight" 
# python3 train_1-2.py ${1} ${2} --mode="test" --model_option="B" --batch_size=1  --checkpth="ckpt_1-2B-2" 

# C
#python3 train_1-2.py ${1} ${2} --mode="test" --model_option="C" --batch_size=1  --checkpth="ckpt_1-2C" 
python3 train_1-2.py ${1} ${2} --mode="test" --model_option="C" --batch_size=1  --checkpth="ckpt_1-2-deeplab"  
#python3 train_1-2.py ${1} ${2} --mode="test" --model_option="C" --batch_size=1  --checkpth="ckpt_1-2-deeplab_new"  
