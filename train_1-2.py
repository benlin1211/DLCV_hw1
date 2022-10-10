import os
import argparse
from re import S
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image
from collections import Counter

from mean_iou_evaluate import mean_iou_score
from viz_mask import viz_data
import imageio


#  NAN indicates there is no pixel predicted as that class in the validation dataset
#  0 indicates there is some pixel predicted as that class, but none of them are correctly placed.

def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

voc_cls = {'urban':0, 
           'rangeland': 2,
           'forest':3,  
           'unknown':6,  
           'barreb land':5,  
           'Agriculture land':1,  
           'water':4} 

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def read_masks(seg, shape):
    masks = np.zeros((1,shape[1],shape[2]))
    mask = seg #(seg >= 128).astype(int)
    mask = 4 * mask[0, :, :] + 2 * mask[1,:, :] + mask[2,:, :]
    mask = np.expand_dims(mask, axis=0)

    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown

    return masks       

# Datalaoder:  199X_sat.jpg (input) / 199X_mask.png (GTH)
# Size: input: 512x512, output: 512x512
class ImageDataset(Dataset):
    def __init__(self, path, tfm, mode):
        super(ImageDataset).__init__()
        self.path = path
        self.files_input = sorted( [os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")], key=lambda x: (int(x.split('/')[-1].split('_')[0])) )
        # for file in self.files_input:
        #     print(file)
        # Ref: https://stackoverflow.com/questions/54399946/python-glob-sorting-files-of-format-int-int-the-same-as-windows-name-sort

        self.files_gth = sorted( [os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")], key=lambda x: (int(x.split('/')[-1].split('_')[0])) )
        
        #for file in self.files_gth:
        #    print(file)

        self.transform = tfm
        self.mode = mode

    def __len__(self):
        return len(self.files_input)
  
    def __getitem__(self,idx):
        fname_input = self.files_input[idx] 
        img_input = Image.open(fname_input)
        img_input = self.transform(img_input)
        # print(fname_input)

        if self.mode == "train":
            fname_gth = self.files_gth[idx] 
            img_gth = Image.open(fname_gth)
            img_gth = self.transform(img_gth)
            img_gth = read_masks(img_gth, img_gth.shape)
            #print(fname_gth)
            ## TODO 把gth改成single value 
            img_gth = torch.tensor(img_gth, dtype=torch.int64)
            return img_input, img_gth

        elif self.mode == "test":
            # print(" test has no label")
            return img_input



def l2_regularizer(model):
    return sum(p.pow(2).sum() for p in model.parameters())


class FocalLoss(nn.Module):
    r"""
        # https://zhuanlan.zhihu.com/p/28527749
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
def train(model, criterion, optimizer, train_loader, batch_size, device, lamb, model_option):
    # Make sure the model is in train mode before training
    model.train()
    
    # These are used to record information in training.
    train_loss = []
    train_pred = np.empty((len(train_loader)*batch_size, 512, 512))
    train_labels = np.empty((len(train_loader)*batch_size, 512, 512))

    pbar = tqdm(train_loader)
    i = 0
    for batch in pbar:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #print("imgs",imgs.shape) #torch.Size([1, 1, 512, 512])
        #print("labels",labels.shape) #torch.Size([1, 512, 512])

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        #print(logits)
        #print("logits",logits.shape) #torch.Size([1, 7, 512, 512])
    
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits.reshape(batch_size, 7, -1), labels.reshape(batch_size, -1).to(device)) + lamb * l2_regularizer(model)
        # print("single loss:", loss)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Record the loss and accuracy.
        train_loss.append(loss.item())

        i=i+batch_size
        pbar.set_description("Loss %.4lf |" % loss)

    # The average loss and accuracy for entire training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)

    #train_mIoU = mean_iou_score(train_pred, train_labels) # return iou 
    train_mIoU = None

    return train_loss, train_mIoU

    
def validate(model, criterion, valid_loader, batch_size, device, lamb, model_option):
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    
    valid_loss = []
    valid_pred = np.empty((len(valid_loader)*batch_size, 512, 512))
    valid_labels = np.empty((len(valid_loader)*batch_size, 512, 512))

    # Iterate the validation set by batches.
    pbar = tqdm(valid_loader)
    i=0
    for batch in pbar:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        logits = model(imgs.to(device))

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits.reshape(batch_size, 7, -1), labels.reshape(batch_size, -1).to(device)) + lamb * l2_regularizer(model)
        
        # Record the loss and accuracy.
        valid_loss.append(loss.item())


        i=i+batch_size
        pbar.set_description("Loss %.4lf |" % loss)
        

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)

    # valid_mIoU = mean_iou_score(valid_pred, valid_labels) # return iou 
    valid_mIoU = None

    return valid_loss, valid_mIoU

# Ref: https://blog.csdn.net/qq_37923586/article/details/106843736
# Model A
class VGG16_FCN32(nn.Module): 
    def __init__(self, num_freeze_layer=0):
        super(VGG16_FCN32, self).__init__()
        #self.feather_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # https://blog.csdn.net/qq_37923586/article/details/106843736
        # https://stackoverflow.com/questions/66085134/get-some-layers-in-a-pytorch-model-that-is-not-defined-by-nn-sequential
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
        # ** https://github.com/sairin1202/fcn32-pytorch/blob/master/pytorch-fcn32.py **
        # https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
        # Ref: https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn32s.py
        self.vgg = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1 ).children())[:-2])

        self.conv1=nn.Sequential(nn.Conv2d(512,4096,2), # nn.Conv2d(512,4096,7) ㄉ
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                )
        # self.conv2=nn.Sequential(nn.Conv2d(4096,4096,2), # nn.Conv2d(4096,4096,1)
        #                         nn.ReLU(inplace=True),
        #                         nn.Dropout()
        #                         )
        self.score_fr=nn.Conv2d(4096,7,1) # num_classes = 7 
        self.upscore = nn.ConvTranspose2d(7, 7, 64, stride=32) # num_classes = 7 
        # self.conv=nn.Sequential(nn.Conv2d(512,7,1),
        #                         nn.ReLU(inplace=True),
        #                         nn.Dropout()
        #                         )
        # self.upsampling = nn.Upsample(scale_factor=32, mode='bilinear')

        
    def forward(self, x):
        #print("fffff")
        #print("input",x.shape)
        out = self.vgg(x)
        #print("vgg",out.shape)

        out = self.conv1(out)
        #print("conv1",out.shape)
        #out = self.conv2(out)
        #print("conv2",out.shape)
        out = self.score_fr(out)
        #print("score_fr",out.shape)
        score = self.upscore(out)
        #print("upscore",score.shape)

        # out = self.conv(out)
        # #print("conv",out.shape)  
        # score = self.upsampling(out)
        # #print("upsampling",out.shape)   

        return score

# Ref: https://blog.csdn.net/qq_37923586/article/details/106843736
# Model B
class VGG16_FCN8s(nn.Module): 
    def __init__(self, num_freeze_layer=0):
        super(VGG16_FCN8s, self).__init__()
        #self.feather_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.vgg = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1 ).children())[:-2])

        # Ref: https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 7, kernel_size=1) # n_classes = 7

        
    def forward(self, x):
        #print("input", x.shape)
        #https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        for idx, layer in enumerate(*list(self.vgg.children())):
            #print(idx)
            x = layer(x)
            if idx == 30: # size=(N, 512, x.H/32, x.W/32)
                x5 = x  
                #print("x5", x5.shape)
            elif idx == 23: # size=(N, 512, x.H/16, x.W/16) #more important
                x4 = x
                #print("x4", x4.shape)
            elif idx == 16: # size=(N, 256, x.H/8,  x.W/8) #most important
                x3 = x
                #print("x3", x3.shape)

        #print("vgg", x.shape)
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + 1*x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        #print("deconv1", score.shape)

        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + 1*x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        #print("deconv2", score.shape)

        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        #print("deconv3", score.shape)

        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        #print("deconv4", score.shape)

        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        #print("deconv5", score.shape)
        
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

# Model C
# https://medium.com/image-processing-and-ml-note/deeplabv3-atrous-convolution-semantic-segmentation-e8cbc111c792
class DeepLabV3_Resnet50(nn.Module): 
    def __init__(self):
        super(DeepLabV3_Resnet50, self).__init__()
        self.model = deeplabv3_resnet50(weight=DeepLabV3_ResNet50_Weights)
        self.model.classifier[4] = nn.Conv2d(256, 7, kernel_size=1, stride=1)


    def forward(self, x):
        score = self.model(x)['out']
        #print("score",score.shape)


        return score
"""
class Resnet50_FCN8s(nn.Module): 
    def __init__(self, num_freeze_layer=0):
        super(Resnet50_FCN8s, self).__init__()
        #self.feather_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
       
        #self.resnet = nn.Sequential(*list( resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2])
        self.resnet = nn.Sequential(*list( resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-2])

        # Ref: https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 7, kernel_size=1) # n_classes = 7

        
    def forward(self, x):
        #print("input", x.shape)
        #https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        for idx, layer in enumerate(self.resnet.children()):
            #print(idx)
            x = layer(x)
            #print(f"idx{idx}", x.shape)
            if idx == 7: # size=(N, 512, x.H/32, x.W/32)
                x5 = x  
                #print("x5", x5.shape)
            elif idx == 6: # size=(N, 512, x.H/16, x.W/16) #more important
                x4 = x
                #print("x4", x4.shape)
            elif idx == 5: # size=(N, 256, x.H/8,  x.W/8) #most important
                x3 = x
                #print("x3", x3.shape)

        #print("resnet", x.shape)
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        #print("deconv1", score.shape)
        score = self.bn1(score + 1*x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        #print("bn1", score.shape)

        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        #print("deconv2", score.shape)
        score = self.bn2(score + 1*x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        #print("bn2", score.shape)

        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        #print("deconv3", score.shape)

        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        #print("deconv4", score.shape)

        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        #print("deconv5", score.shape)
        
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        #print("classifier", score.shape)

        return score  # size=(N, n_class, x.H/1, x.W/1)
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Training data location")
    parser.add_argument("dest", help="Image prediction output location (for test mode)")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--checkpth", help="Checkpoint location", default = "ckpt_seg")
    parser.add_argument("--batch_size", help="batch size", type=int, default=5)
    parser.add_argument("--model_option", help="Choose \"A\" or \"B\". (CNN from scratch or Resnet)", default="A")
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.0)
    parser.add_argument("--scheduler_lr_decay_step", help="scheduler learning rate decay step ", type=int, default=1)
    parser.add_argument("--scheduler_lr_decay_ratio", help="scheduler learning rate decay ratio ", type=float, default=0.99)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=40)
    parser.add_argument("--n_split", help="k-fold split numbers", type=int, default=5)      
    parser.add_argument("--l2_reg_lambda", help="Lambda value for L2 regularizer", type=float, default=0.0001)   
    args = parser.parse_args()
    print(vars(args))
    
    src_path = args.src
    des_path = args.dest
    mode = args.mode
    model_path = args.checkpth
    batch_size = args.batch_size
    model_option = args.model_option
    lr = args.learning_rate
    weight_decay = args.weight_decay
    lr_decay_step = args.scheduler_lr_decay_step
    lr_decay_ratio = args.scheduler_lr_decay_ratio
    
    n_epochs = args.n_epochs
    n_split = args.n_split
    l2_lamb = args.l2_reg_lambda

    # fix random seed
    fix_random_seed()

    # GPU
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"device = {device}")
    
    
    train_tfm = transforms.Compose([
        transforms.Resize((512, 512)), # Upsampling
        # best: no ColorJitter
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
    ])
   
    # train
    if mode == "train":

        # Load dataset
        dataset = ImageDataset(src_path, tfm=train_tfm, mode=mode)
        kfold = KFold(n_splits=n_split, shuffle=True)
        print(len(dataset))

        # loss
        criterion = nn.CrossEntropyLoss()
        
        for i, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

            train_set = Subset(dataset, train_ids)
            valid_set = Subset(dataset, val_ids)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
            # model
            if model_option == "A":
                print("A: VGG16 + FCN32")
                model = VGG16_FCN32().to(device)
            elif model_option == "B":
                print("B: VGG16 + FCN8s")
                model = VGG16_FCN8s().to(device)
            elif model_option == "C":
                #print("C: Resnet50 + FCN8s")
                #model = Resnet50_FCN8s().to(device)
                print("C: DeepLabV3 + Resnet50")
                model = DeepLabV3_Resnet50().to(device)
                #model = deeplabv3_resnet50(weight=DeepLabV3_ResNet50_Weights).to(device)
                #print(model)
                #model.classifier._module['4'] = nn.Conv2d(512, 7, kernel_size=1, stride=1)
                #model.aux_classifier._module['4'] = nn.Conv2d(256, 7, kernel_size=1, stride=1)

                #print(model.classifier)
                #print(model.classifier._module['4'])
                #print(model.aux_classifier) 
                #print(model.aux_classifier._module['4'])
            print(model)
            if os.path.exists(os.path.join( model_path, f"hw1-2-{model_option}_fold{i}.ckpt") ):
                prev = os.path.join( model_path, f"hw1-2-{model_option}_fold{i}.ckpt") 
                print(f"Loading previous checkpoint: { prev }")
                model.load_state_dict(torch.load(prev))
            
            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_ratio)    

            best_loss = 1e+8
            # Training loop
            for epoch in range(n_epochs):
                
                print("Fold:",i)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                
                # ---------- Training ----------
                train_loss, train_mIoU = train(model, criterion, optimizer, train_loader, batch_size, device, l2_lamb, model_option)
                print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
                scheduler.step()
                    
                # ---------- Validation ----------
                print("Start validation.")
                valid_loss, valid_mIoU = validate(model, criterion, valid_loader, batch_size, device, l2_lamb, model_option)
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")

                # update logs
                if valid_loss < best_loss:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f} -> best")
                else:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")

                # save models for report
                if epoch == 0 or  epoch == int(n_epochs/2) or epoch == n_epochs-1:
                    print(f"Saving stage model at {model_path}")
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(),  os.path.join( model_path, f"hw1-2-{model_option}_epoch{epoch}_fold{i}.ckpt") ) 

                # save best models    
                if valid_loss < best_loss:
                    print(f"Best model found at epoch {epoch}, saving model at {model_path}")
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(),  os.path.join( model_path, f"hw1-2-{model_option}_fold{i}.ckpt") ) 
                    best_loss = valid_loss

                # update epoch record
                epoch = epoch + 1  
            # if i == 0:
            #     print("Training is done.")
            #     break
                


    # test
    print("Mode:",mode)
    if mode == "test":  

        cmap = cls_color
        if not os.path.exists(des_path):
            os.makedirs(des_path)

        # Load dataset
        dataset = ImageDataset(src_path, tfm=train_tfm, mode=mode)
        num_tests = len(dataset)

        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_pred = np.zeros((num_tests, 512, 512))

        for i in range(n_split):
            print(f"fold {i}:")
            if model_option == "A":
                print("A: VGG16 + FCN32")
                model = VGG16_FCN32().to(device)
            elif model_option == "B":
                print("B: VGG16 + FCN8s")
                model = VGG16_FCN8s().to(device)
            elif model_option == "C":
                #print("C: Resnet50 + FCN8s")
                #model = Resnet50_FCN8s().to(device)
                print("C: DeepLabV3 + Resnet50")
                model = DeepLabV3_Resnet50().to(device)

            ckpt_name = f"hw1-2-{model_option}_fold{i}.ckpt"
            #ckpt_name = f"hw1-2-{model_option}_epoch0_fold{i}.ckpt"
            print(f"Loading checkpoint {os.path.join(model_path, ckpt_name)}.")
            model.load_state_dict(torch.load( os.path.join(model_path, ckpt_name)))
            
            model.eval()
            print("Predicting...")
            with torch.no_grad():
                j=0
                pbar = tqdm(test_dataloader)
                for data in pbar:
                    #print("data",data)
                    logits = model(data.to(device))
                    #print("logits",logits.shape)
                    # test_pred[j] = np.argmax(logits.detach().cpu(), axis=1)
                    pred = np.argmax(logits.detach().cpu(), axis=1).reshape(512,512)
                    #print("pred",pred.shape)

                    # get prediction kinds
                    cs = np.unique(pred)
                    #print(j, cs)
                    #color_masks = np.zeros((len(cs),512, 512, 3))
                    result_img = np.zeros((512*512, 3))

                    for k,c in enumerate(cs):
                        # print( cmap[c] )
                        mask = np.zeros((512, 512))
                        ind = np.where(pred==c) # pred.shape: 1,512,512
                        #print(ind)
                        mask[ind[0], ind[1]] = 1

                        # img = viz_data(img, mask, color=cmap[c])
                        l_loc = np.where(mask.flatten() == 1)[0]
                        #print(l_loc.shape)

                        # Unknown: 先猜農地
                        if cmap[c] == [0, 0, 0]: 
                            #print(f"Unknown detected at img {j}.")
                            #print(len(l_loc))
                            result_img[l_loc, : ] = [255, 255, 0]
                        else:
                            result_img[l_loc, : ] = cmap[c]
                        # result_imgs: 512x512, 3. bg: [0,0,0] fg: cmap[c]

                    result_img = result_img.reshape((512, 512, 3))  
                    imageio.imsave(os.path.join(des_path, "{:04d}_mask.png".format(j)), np.uint8(result_img))
                    j=j+1
                    
            if i == 0:
                print("Testing is done. (one fold)")
                break            

        # Convert prediction(1,512,512) to RGB image(512,512,3)
        #test_pred_imgs = np.empty((num_tests, 512, 512, 3))
        #pred_imgs = np.empty((512, 512, 3))


        """
        i=0
        print("Generating images...")
        for pred in tqdm(test_pred):
            # get prediction kinds
            cs = np.unique(pred)
            img = np.zeros((512, 512, 3))

            #print(i, cs)
            color_masks = np.zeros((len(cs),512, 512, 3))
            for k,c in enumerate(cs):
                # print( cmap[c] )
                mask = np.zeros((512, 512))
                ind = np.where(pred==c)
                mask[ind[0], ind[1]] = 1

                # img = viz_data(img, mask, color=cmap[c])
                color_mask = np.zeros((512*512, 3))
                l_loc = np.where(mask.flatten() == 1)[0]
                color_mask[l_loc, : ] = cmap[c]
                # color_mask: 512x512, 3. bg: [0,0,0] fg: cmap[c]
                color_mask = color_mask.reshape((512, 512, 3))  
                #print(color_mask)

                # overlap the predict classes
                # assumption: we only have one label per pixel. 
                color_masks[k] = color_mask 

            for color_mask in color_masks:
                img = img + color_mask ##?

            imageio.imsave(os.path.join(des_path, "{:04d}_mask.png".format(i)), np.uint8(img))
            #print( "{:04d}_mask.png".format(i))
            i = i+1
        print(f"Images are generated at {des_path}")
        """
        # ensembling    
        # prediction_final = []
        # for i in range(n_split):
        #     print(len( predictions[i]))
        # for j in range(num_tests):
        #     vote_box = []
        #     for i in range(n_split):
        #         #print(predictions[i][j])
        #         vote_box.append(predictions[i][j])
        #     counts = Counter(vote_box)
        #     # get the frequency of the most.
        #     max_count = counts.most_common(1)[0][1] 
        #     # get the result that equals to that frequency.
        #     out = [value for value, count in counts.most_common() if count == max_count]
        #     # draw:
        #     if len(out)>1: 
        #         # flip to decide...
        #         out = [random.choice(out)]
        #     # turn list into single value
        #     # print(f"==={j}=== out:",out)
        #     out = out[0]
        #     prediction_final.append(out)
        # #print(prediction_final)

