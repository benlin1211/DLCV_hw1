#!/bin/bash

python3 download_model.py
# https://github.com/wkentaro/gdown/issues/163
unzip -o ckpt.zip 
rm -rf ckpt.zip 

unzip -o ckpt1-2.zip
rm -rf ckpt1-2.zip
