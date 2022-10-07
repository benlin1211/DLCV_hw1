import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image

from collections import Counter



def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    pred = pred.numpy()
    labels = labels.numpy()
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp + 1e-8) # Avoid NaN
        mean_iou += iou / 6
        #print(f"tp_fp:{tp_fp}, tp_fn:{tp_fn}, tp:{tp}, tp_fp + tp_fn - tp:{tp_fp + tp_fn - tp}")
        #print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


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

        elif self.mode == "test":
            img_gth = None 
            # print(" test has no label")
        return img_input, img_gth


def l2_regularizer(model):
    return sum(p.pow(2).sum() for p in model.parameters())


def train(model, criterion, optimizer, train_loader, batch_size, device, lamb = 0.001):
    # Make sure the model is in train mode before training
    model.train()
    
    # These are used to record information in training.
    train_loss = []
    train_mean_iou = []
    
    pbar = tqdm(train_loader)
    for batch in pbar:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #print("imgs",imgs.shape)
        #print("labels",labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        
        labels = labels.reshape(batch_size, -1)
        logits = logits.reshape(batch_size, 7, -1)
        #print("labels",labels.shape)
        #print("predict",predict.shape)

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device)) + lamb * l2_regularizer(model)
        # print("single loss:", loss)
        # myLoss = customCrossEntropy(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute tp_fn, tp_fp, fn for current batch.

        # Record the loss and accuracy.
        train_loss.append(loss.item())

        # Compute the accuracy for current batch.\
        pred = torch.argmax(logits, dim=1).clone().detach().cpu()
        mean_IoU = mean_iou_score(pred, labels.clone().detach().cpu())
        train_mean_iou.append(mean_IoU)

        pbar.set_description("Loss %.4lf, mIoU %.4lf" % (loss, mean_IoU))
        # train_accs.append(acc)

    # The average loss and accuracy for entire training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    # TODO: change variable name of train_acc
    train_acc = sum(train_mean_iou) / len(train_mean_iou) # return iou 

    return train_loss, train_acc

    
def validate(model, criterion, valid_loader, batch_size, device, lamb):
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    
    valid_loss = []
    valid_mean_iou = []

    # Iterate the validation set by batches.
    pbar = tqdm(valid_loader)
    for batch in pbar:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        logits = model(imgs.to(device))

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))
            labels = labels.reshape(batch_size, -1)
            logits = logits.reshape(batch_size, 7, -1)

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device)) + lamb * l2_regularizer(model)

        # Record the loss and accuracy.
        valid_loss.append(loss.item())

        # Compute the accuracy for current batch.
        pred = torch.argmax(logits, dim=1).clone().detach().cpu()
        mean_IoU = mean_iou_score(pred, labels.clone().detach().cpu())
        valid_mean_iou.append(mean_IoU)

        pbar.set_description("Loss %.4lf, mIoU %.4lf" % (loss, mean_IoU) )
        # train_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    # TODO: change variable name of valid_acc
    valid_acc = sum(valid_mean_iou) / len(valid_mean_iou) 

    return valid_loss, valid_acc

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

        self.conv1=nn.Sequential(nn.Conv2d(512,4096,2), # nn.Conv2d(512,4096,7)
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                )

        self.conv2=nn.Sequential(nn.Conv2d(4096,4096,1),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )
        self.score_fr=nn.Conv2d(4096,7,1) # num_classes = 7 
        self.upscore = nn.ConvTranspose2d(7, 7, 64, stride=32) # num_classes = 7 

    def forward(self, x):
        #print("fffff")
        #print(x.shape)
        out = self.vgg(x)
        #print(out.shape)
        out = self.conv1(out)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.score_fr(out)
        #print(out.shape)
        score = self.upscore(out)
        #print(score.shape)

        return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Training data location")
    parser.add_argument("dest", help="CSV prediction output location (for test mode)")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--checkpth", help="Checkpoint location", default = "ckpt_seg")
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument("--model_option", help="Choose \"A\" or \"B\". (CNN from scratch or Resnet)", default="A")
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.0)
    parser.add_argument("--scheduler_lr_decay_step", help="scheduler learning rate decay step ", type=int, default=1)
    parser.add_argument("--scheduler_lr_decay_ratio", help="scheduler learning rate decay ratio ", type=float, default=0.99)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=50)
    parser.add_argument("--n_split", help="k-fold split numbers", type=int, default=5)    
    parser.add_argument("--patience", help="Training patience", type=int, default=10)   
    parser.add_argument("--l2_reg_lambda", help="Lambda value for L2 regularizer", type=float, default=0.001)   
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
    patience = args.patience
    l2_lamb = args.l2_reg_lambda

    # fix random seed
    fix_random_seed()

    # GPU
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    
    
    train_tfm = transforms.Compose([
        transforms.Resize((512, 512)), # Upsampling
        # best: no ColorJitter
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
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
                print("B: ???")
                model = None
            print(model)

            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_ratio)    

            stale = 0 # count for patiency
            best_acc = 0
            # Training loop

            for epoch in range(n_epochs):
                
                print("Fold:",i)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                
                # ---------- Training ----------
                train_loss, train_acc = train(model, criterion, optimizer, train_loader, batch_size, device, lamb=l2_lamb)
                print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
                scheduler.step()
                    
                # ---------- Validation ----------
                valid_loss, valid_acc = validate(model, criterion, valid_loader, batch_size, device, lamb=l2_lamb)
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                # update logs
                if valid_acc > best_acc:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
                else:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                # save models
                if valid_acc > best_acc:
                    print(f"Best model found at epoch {epoch}, saving model")
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(),  os.path.join( model_path, "hw1-2-{model_option}_fold{i}.ckpt") ) 
                    best_acc = valid_acc
                    stale = 0
                else:
                    stale += 1
                    print(f"No improvment {stale}")
                    if stale > patience:
                        print(f"No improvment {patience} consecutive epochs, early stopping")
                        break

                # update epoch record
                epoch = epoch + 1  



    # test
    print("Mode:",mode)
    if mode == "test":

        # Load dataset
        dataset = ImageDataset(src_path, tfm=train_tfm, mode=mode)
        num_tests = len(dataset)
        print(num_tests)

        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = [ [] for _ in range(n_split) ]

        # get filenames
        filename_list = []
        for _, filename in tqdm(test_dataloader):
            filename_list += filename

        for i in range(n_split):
            print(f"fold {i}:")
            if model_option == "A":
                print("A: VGG16 + FCN32")
                model = VGG16_FCN32().to(device)
            elif model_option == "B":
                print("B: Resnet")
                model = None

            model.load_state_dict(torch.load( os.path.join( model_path, "hw1-2-{model_option}_fold{i}.ckpt")))
            model.eval()
            
            with torch.no_grad():
                j=0
                for data, _ in tqdm(test_dataloader):
                    test_pred = model(data.to(device))
                    test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                    predictions[i] += test_label.squeeze().tolist()
                    # j=j+1
                    # if j>2:
                    #     break

        # ensembling    
        prediction_final = []
        for i in range(n_split):
            print(len( predictions[i]))
        for j in range(num_tests):
            vote_box = []
            for i in range(n_split):
                #print(predictions[i][j])
                vote_box.append(predictions[i][j])
            counts = Counter(vote_box)
            # get the frequency of the most.
            max_count = counts.most_common(1)[0][1] 
            # get the result that equals to that frequency.
            out = [value for value, count in counts.most_common() if count == max_count]
            # draw:
            if len(out)>1: 
                # flip to decide...
                out = [random.choice(out)]
            # turn list into single value
            # print(f"==={j}=== out:",out)
            out = out[0]
            prediction_final.append(out)
        #print(prediction_final)

        # write_result
        df = pd.DataFrame() # apply pd.DataFrame format 
        df["filename"] = [x.split('/')[-1] for x in filename_list]
        df["label"] = prediction_final
        if not os.path.exists(des_path):
            os.makedirs(des_path)
        df.to_csv(os.path.join(des_path, f"val_{model_option}.csv"),index = False)


