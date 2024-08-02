import argparse

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
import random

import torch
import torch.nn as nn # edited
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b3, EfficientNet_B3_Weights

from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

# Library bugfix: it is necessary to override the get_state_dict method in order to 
# use the pretrained weights from the torch.hub
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

NUM_CLASSES = 10

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def validate(args, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    val_losses = []
    val_acc = 0.0
    total = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data_loader, desc="Validation")):
            image = x.to(args.device)
            label = y.to(args.device)
            
            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            
            val_losses.append(loss.item())
            total += label.size(0)

            val_acc += acc(output, label)

    epoch_val_loss = np.mean(val_losses)
    epoch_val_acc = val_acc / total
    
    print(f'Validation Loss: {epoch_val_loss}')
    print('Validation Accuracy: {:.3f}'.format(epoch_val_acc * 100))
    
    return epoch_val_loss, epoch_val_acc

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(args, data_loader, val_loader, model):
    """
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max')
    
    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)

            optimizer.zero_grad()

            if np.random.rand(1) < args.p_mixup:
                mixed_image, label_a, label_b, lam = mixup_data(image, label)
                output = model(mixed_image)
                loss = mixup_criterion(criterion, output, label_a, label_b, lam)
            else:
                output = model(image)
                label = label.squeeze()
                loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))

        # Validation after every epochs
        epoch_val_loss, epoch_val_acc = validate(args, val_loader, model)

        scheduler.step(epoch_val_acc)
        
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = NUM_CLASSES
    
    """
    """
    
    # hyperparameters
    args.epochs = 80
    args.learning_rate = 0.1
    args.batch_size = 64
    args.p_mixup = 0.5

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, val_loader = make_data_loader(args)

    # custom model
    # model = BaseModel()
    
    # torchvision model
    #model = resnet18(weights=ResNet18_Weights)
    model = efficientnet_b3(weights=EfficientNet_B3_Weights)
    
    # you have to change num_classes to 10
    #num_features = model.fc.in_features # edited
    num_features = model.classifier[1].in_features
    #model.fc = nn.Linear(num_features, num_classes) # edited
    model.classifier = nn.Linear(num_features, num_classes)
    model.to(device)
    print(model)

    # Training The Model
    train(args, train_loader, val_loader, model)