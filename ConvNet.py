import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import CancerDataset
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input = 96 x 96 x 3
        #self.conv1 = nn.Conv2d(3, 32, 5, bias=True)
        #self.conv2 = nn.Conv2d(32, 64, 3, bias=True)
        #self.conv3 = nn.Conv2d(64, 128, 2, bias=True)
        self.conv1 = nn.Conv2d(3, 16, 3, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*10*10, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        #self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize dataset
csvfile = '/Users/bmmidei/Projects/CancerDetection/data/train_labels.csv'
dataPath = '/Users/bmmidei/Projects/CancerDetection/data/train'
ds = CancerDataset(csv_file=csvfile , image_dir=dataPath)

dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

# Train the model
num_epochs = 1
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, data in enumerate(dl):
        im_batch = data['image']
        label_batch = data['label']

        # Run the forward pass
        outputs = model(im_batch)
        loss = criterion(outputs, label_batch)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = label_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label_batch).sum().item()
        acc_list.append(correct / total)
        if (i + 1) % 2 == 0:
            print(loss.item())
            '''
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
            '''

