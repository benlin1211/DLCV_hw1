import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random


class ImageDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super(ImageDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx] 
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
            # print(label)
        except:
            label = -1 # test has no label
            # print(" test has no label")
        return im,label





def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Training data location")
    parser.add_argument("checkpth", help="Checkopint location")
    args = parser.parse_args()
    # config = vars(args)


    print("hello")
    fix_random_seed()

    print(args.src)
    print(args.checkpth)