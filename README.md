
Please click [this link](https://docs.google.com/presentation/d/1lXkZrUrV209kMSGn6Lg37rno0Kp_zbdyxOl0K8F9U_E/edit?usp=sharing) to view the slides of HW1.

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2022/hw1-<username>.git
Note that you should replace `<username>` with your own GitHub username.

# Submission Rules
### Deadline
2022/10/10 (Mon.) 23:59


### Packages
This homework should be done using python3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to run virtual environment, and install all the packages listed in the requirements.txt:

    conda create --name DLCV-hw1 python=3.8
    conda activate DLCV-hw1
    pip3 install -r requirements.txt
    
If you have 2 CUDA: please do the following: (the training is based on CUDA:1)

    export CUDA_VISIBLE_DEVICES=0,1

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### List all environments
    conda info --envs

### Close an environment
    conda deactivate

### Remove an environment
    conda env remove -n DLCV-hw1

### Argparse
https://machinelearningmastery.com/command-line-arguments-for-your-python-script/

### Run Training Code

    bash ./train_hw1-1.sh ./hw1_data/hw1_data/p1_data/train_50 .
    bash ./train_hw1-2.sh ./hw1_data/hw1_data/p2_data/train .
    
### Run Inference Code 

    bash ./hw1_1.sh ./hw1_data/hw1_data/p1_data/val_50 result_1-1 
    
#### 1-2A 

    bash ./hw1_2.sh ./hw1_data/hw1_data/p2_data/validation result_1-2

#### 1-2B-1

    bash ./hw1_2.sh ./hw1_data/hw1_data/p2_data/validation result_1-2B
    
#### 1-2B-2

    bash ./hw1_2.sh ./hw1_data/hw1_data/p2_data/validation result_1-2B_no_weight
    
### Acccuracy

    bash ./eval_1-1.sh "./val_A.csv" "./hw1_data/hw1_data/p1_data/val_gt.csv"
    bash ./eval_1-1.sh "./val_B.csv" "./hw1_data/hw1_data/p1_data/val_gt.csv"
    bash ./eval_1-2.sh "./result_1-2" "./hw1_data/hw1_data/p2_data/validation"
    bash ./eval_1-2.sh "./result_1-2B" "./hw1_data/hw1_data/p2_data/validation"
    bash ./eval_1-2.sh "./result_1-2B_no_weight" "./hw1_data/hw1_data/p2_data/validation"

# Q&A
If you have any problems related to HW1, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw1 FAQ section in FB group.(But TAs won't answer your question on FB.)
