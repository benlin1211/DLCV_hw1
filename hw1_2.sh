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
# update-client 2>&1 | tee training_log.txt
# exec > training_log2.txt

# for ((i=0; i<22; i++)); do

#     python3 train_1-2.py ${1} ${2} --mode="test" --model_option="C" --batch_size=1  --checkpth="ckpt_1-2-deeplab-exp2" --resume_n=${i}
#     # mIoU calculation
#     bash ./eval_1-2.sh ${2} ${1}

# done
# bash ./eval_1-2.sh ${2} ${1}

# for demo
python3 train_1-2.py ${1} ${2} --mode="test" --model_option="C" --batch_size=1  --checkpth="ckpt_1-2-ensemble" 

