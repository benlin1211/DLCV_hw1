#!/bin/bash

python3 download_model.py
# https://github.com/wkentaro/gdown/issues/163
unzip -o ckpt.zip 
rm -rf ckpt.zip 


