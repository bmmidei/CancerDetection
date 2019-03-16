import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from utils import CancerDataset

import pandas as pd
import numpy as np
import math, os
from PIL import Image

sample_sub = '../input/sample_submission.csv'
test_csv = '../input/sample_submission.csv'

train_csv = '../input/train_labels.csv'

train_dir = '../input/train/'
test_dir = '../input/test/'


class Net(nn.Module):

    '''Network class to construct 3-layer ConvNet'''
    def __init__(self):
        super(Net, self).__init__()
        # input = 96 x 96 x 3
        # Define network layers with correct shapes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64*10*10, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Create forward pass of convolutional layers with the structure:
        # Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn3(x)))

        # Reshape to a flattened layer and pass through Dense layers
        x = x.view(-1, 64*10*10)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train(model, train_dl, batch_size, num_epochs, learning_rate=0.001):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    print("***** Model Parameters *****")
    print("batch_size=", batch_size)
    print("epochs=", num_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 28)

    for epoch in range(num_epochs):
        # Iterate through data loader
        for i, data in enumerate(train_dl):
            im_batch = data['image']
            label_batch = data['label']

            # Zero gradient before forward pass
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(im_batch)
            loss = criterion(outputs, label_batch)
            

            # Compute Gradients and update parameters with Adam Optimizer
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(loss.item())

train_df = pd.read_csv(train_csv)
im_transform = transforms.Compose([
                  transforms.RandomHorizontalFlip(), 
                  transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(20),
                  transforms.ToTensor()])

# Initialize dataset
ds = CancerDataset(train_df, image_dir=train_dir, transform=im_transform)

num_epochs = 3
batch_size = 64

# Split training set into train and test
# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

# Device config for GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

train(model, train_dl, batch_size, num_epochs, 0.001)
test_df = pd.read_csv(test_csv)

# Create test dataset - Note that this is NOT shuffled
test_ds = CancerDataset(test_df, image_dir=test_dir, transform=None)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

# Evaluate using moving mean and variance from batch norm
model.eval()
with torch.no_grad():
    preds = []
    for i, data in enumerate(test_dl):
        im_batch = data['image']

        # Run the forward pass to generate predictions
        outputs = model(im_batch)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.detach().cpu().numpy()
        for i in predicted:
            preds.append(i)

print('{} predictions generated'.format(len(preds)))

sub_df = pd.read_csv(sample_sub)
sub_df.drop('label', axis=1)
sub_df['label'] = preds
sub_df.to_csv('submission.csv', index=False)