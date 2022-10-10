#!/bin/bash

# TODO - run report question
python3 viz_mask.py --seg_path "./result_1-2C/0013_mask.png" --img_path "hw1_data/hw1_data/p2_data/validation/0013_sat.jpg" --save_as "0013.jpg"
python3 viz_mask.py --seg_path "./result_1-2C/0062_mask.png" --img_path "hw1_data/hw1_data/p2_data/validation/0062_sat.jpg" --save_as "0062.jpg"
python3 viz_mask.py --seg_path "./result_1-2C/0104_mask.png" --img_path "hw1_data/hw1_data/p2_data/validation/0104_sat.jpg" --save_as "0104.jpg"
    


