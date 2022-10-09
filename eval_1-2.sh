#!/bin/bash

# TODO - run your inference Python3 code

#for ((i=0; i<=$#; i++)); do
#  echo "parameter $i --> ${!i}"
#done

python3 mean_iou_evaluate.py -p ${1} -g ${2} 
