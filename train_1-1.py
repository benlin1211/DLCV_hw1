import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas
from PIL import Image

def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        

class ImageDataset(Dataset):
    def __init__(self, path, tfm, files = None):
        super(ImageDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")])
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx] 
        im = Image.open(fname)
        # 32x32
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
            # print(label)
        except:
            label = -1 # test has no label
            # print(" test has no label")
        return im,label


def l2_regularizer(model):
    return sum(p.pow(2).sum() for p in model.parameters())

def train(model, criterion, optimizer, train_loader, device, lamb = 0.001):
    
    # Make sure the model is in train mode before training
    model.train()
    
    # These are used to record information in training.
    train_loss = []
    train_accs = []
    
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        # print(labels.shape)
        
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

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
    
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc

    
def validate(model, criterion, valid_loader, device, lamb):
    
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device)) + lamb * l2_regularizer(model)
        
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc


# Model B
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        # stride and padding: 影響第一維
        # pooling: 縮小每張圖的pixel
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # [64, 128/2, 128/2]
            nn.BatchNorm2d(64),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 2, 1), # [256, 64, 64] 
            nn.BatchNorm2d(256),
        )
        self.cnn3 = nn.Sequential( 
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
        )
        # pooling
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]  
            nn.BatchNorm2d(512),
        )
        # pooling
        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8] 
            nn.BatchNorm2d(512),
        )
        # pooling
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 50)
        )
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(2, 2, 0)
        
    def forward(self, x):
        # print(x.shape)
        out = self.cnn1(x)                  # [64, 128/2, 128/2]
        
        # print(out.shape)
        out = self.cnn2(out)                # [256, 64/2, 64/2] 

        # print(out.shape)
        out = self.cnn3(out) + out          # [256, 32, 32] 
        out = self.maxPool(self.relu(out))  # [256, 16, 16] 
        # print(out.shape) 
        out = self.cnn4(out)                # [512, 16, 16]
        out = self.maxPool(self.relu(out))  # [512, 8, 8]
        # print(out.shape) 
        out = self.cnn5(out) + out          # [512, 8, 8]
        out = self.maxPool(self.relu(out))  # [512, 4, 4]
        # print(out.shape)
        
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Training data location")
    parser.add_argument("checkpth", help="Checkopint location")
    parser.add_argument("--batch_size", help="batch size", default=32)
    parser.add_argument("--model_option", help="Choose \"A\" or \"B\". (from scratch or resnet)", default="B")
    parser.add_argument("--learning_rate", help="learning rate", default=5e-5)
    parser.add_argument("--weight_decay", help="weight decay", default=0)
    parser.add_argument("--scheduler_lr_decay_step", help="scheduler learning rate decay step ", default=3)
    parser.add_argument("--scheduler_lr_decay_ratio", help="scheduler learning rate decay ratio ", default=0.99)
    parser.add_argument("--n_epochs", help="n_epochs", default=50)
    parser.add_argument("--n_split", help="k-fold split numbers", default=5)    
    parser.add_argument("--patience", help="Training patience", default=30)   
    parser.add_argument("--l2_reg_lambda", help="Lambda value for L@ regularizer", default=0.001)   
    args = parser.parse_args()
    print(vars(args))

    src_path = args.src
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.1),
        transforms.RandomRotation(degrees=(-15,+15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = ImageDataset(src_path, tfm=train_tfm)
    kfold = KFold(n_splits=n_split, shuffle=True)
    # print(len(image_dataset))
    # train_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True)

    # loss
    criterion = nn.CrossEntropyLoss()
   
    for i, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

        train_set = Subset(dataset, train_ids)
        valid_set = Subset(dataset, val_ids)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        # model
        model = Classifier().to(device)

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
            train_loss, train_acc = train(model, criterion, optimizer, train_loader, device, lamb=l2_lamb)
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
            scheduler.step()
                
            # ---------- Validation ----------
            valid_loss, valid_acc = validate(model, criterion, valid_loader, device, lamb=l2_lamb)
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # update logs
            if valid_acc > best_acc:
                with open(f"./hw1-1_{model_option}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                with open(f"./hw1-1_{model_option}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.state_dict(), f"hw1-1-{model_option}.ckpt") 
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