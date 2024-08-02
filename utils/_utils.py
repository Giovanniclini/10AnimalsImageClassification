from torchvision import datasets, transforms
import torch
import numpy as np
# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader, Subset

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# you can change input size(don't forget to change linear layer!)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cutout_prob = 0.5
cutout_transform = Cutout(length=24)

# Define data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.RandomApply([cutout_transform], p=cutout_prob)
])

def make_data_loader(args):
    
    # Get Dataset
    dataset = datasets.ImageFolder(args.data)
    
    # split dataset to train/test
    train_data_percentage = 0.8
    train_size = int(train_data_percentage * len(dataset))
    #test_size = len(dataset) - train_size
    
    # you must set "seed" to get same test data
    # you can't compare different test set's accuracy
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    # Apply transforms to the subsets
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = val_transform

    # Get Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader