#!/bin/bash

# TODO - run your inference Python3 code
# Test: generate image one by one.
python3 train_1-2.py ${1} ${2} --mode="test" --model_option="A" --batch_size=1