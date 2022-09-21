#!/bin/bash

# TODO - run your inference Python3 code

#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done

python3 train.py ${1} ${2} 
