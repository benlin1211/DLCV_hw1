import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image

from collections import Counter


def fix_random_seed():
    myseed = 6666  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        

class ImageDataset(Dataset):
    def __init__(self, path, tfm, mode, files = None):
        super(ImageDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")], key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1].split('.')[0]))) # TODO: sort with correct order!
        # Ref: https://stackoverflow.com/questions/54399946/python-glob-sorting-files-of-format-int-int-the-same-as-windows-name-sort

        if files != None:
            self.files = files
        self.transform = tfm
        self.mode = mode

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx] 
        im = Image.open(fname)
        # 32x32
        im = self.transform(im)
        # im = self.data[idx]
        if self.mode == "train":
            label = int(fname.split("/")[-1].split("_")[0])
            # print(label)
        elif self.mode == "test":
            label = fname # return filename instead
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
        #print(labels)
        #print(labels.shape)
        #print(logits)
        #print(logits.shape)
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


# Model A
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        # stride and padding: 影響第一維
        # pooling: 縮小每張圖的pixel
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),                  # [64, 128, 128
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64] 

            nn.BatchNorm2d(128),
            nn.ReLU(),                  # [128, 64, 64]
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32] 
        )
        self.cnn3 = nn.Sequential( 
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),                  # [256, 32, 32] 
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8] 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4] 
        )
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 50)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # print(x.shape)
        out = self.cnn1(x)
        # print(out.shape)
        out = self.cnn2(out)
        # print(out.shape)
        out = self.cnn3(out)
        # print(out.shape) 
        out = self.cnn4(out)
        # print(out.shape) 
        out = self.cnn5(out)
        # print(out.shape) 
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# Model B
class Resnet(nn.Module): 
    def __init__(self, num_freeze_layer=0):
        super(Resnet, self).__init__()
        #self.feather_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feather_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze top layers (if needed)
        for i,child in enumerate(self.feather_extractor.children()):
            if i < num_freeze_layer:
                for param in child.parameters():
                    param.requires_grad = False
                #print(i, "(freezed):", child)
            else:
                pass
                #print(i, ":", child)

        self.classifier = nn.Sequential(
            nn.Linear(1000, 50),
            #nn.ReLU(),
            #nn.Linear(1000, 50),
        )
        

    def forward(self, x):
        out = self.feather_extractor(x)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Training data location")
    parser.add_argument("dest", help="CSV prediction output location (for test mode)")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--checkpth", help="Checkpoint location")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
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
    
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)), # Upsampling
        # best: no ColorJitter
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
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
                print("A: CNN")
                model = CNN().to(device)
            elif model_option == "B":
                print("B: Resnet")
                model = Resnet().to(device)
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
                train_loss, train_acc = train(model, criterion, optimizer, train_loader, device, lamb=l2_lamb)
                print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
                scheduler.step()
                    
                # ---------- Validation ----------
                valid_loss, valid_acc = validate(model, criterion, valid_loader, device, lamb=l2_lamb)
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                # update logs
                if valid_acc > best_acc:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
                else:
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                # save models
                if True:
                    print(f"Saving stage model")
                    if not os.path.exists("ckpt"):
                        os.makedirs("ckpt")
                    torch.save(model.state_dict(), f"./ckpt/hw1-1-{model_option}_epoch{epoch}_fold{i}.ckpt") 

                if valid_acc > best_acc:
                    print(f"Best model found at epoch {epoch}, saving model")
                    if not os.path.exists("ckpt"):
                        os.makedirs("ckpt")
                    torch.save(model.state_dict(), f"./ckpt/hw1-1-{model_option}_fold{i}.ckpt") 
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

            if i == 0:
                print("Training is done.")
                break

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
                print("A: CNN")
                model = CNN().to(device)
            elif model_option == "B":
                print("B: Resnet")
                model = Resnet().to(device)

            model.load_state_dict(torch.load(f"./ckpt/hw1-1-{model_option}_fold{i}.ckpt"))
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


