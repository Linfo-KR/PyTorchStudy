import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        
        return op
    
def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred_prob = model(X)
        loss = F.nll_loss(pred_prob, y)
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()
                )
            )
                
def test(model, device, test_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            
            loss += F.nll_loss(pred_prob, y, reduction='sum').item()
            pred = pred_prob.argmax(dim=1, keepdim=True)
            success += pred.eq(y.view_as(pred)).sum().item()
    
    loss /= len(test_dataloader.dataset)
    print('\nTest dataset: Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss,
        success,
        len(test_dataloader.dataset),
        100. * success / len(test_dataloader.dataset))
    )
        
train_dataloader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1302,), (0.3069,))])),
    batch_size = 1024, shuffle=True
)

test_dataloader = DataLoader(
    datasets.MNIST('./data', train=False,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1302,), (0.3069,))])),
    batch_size = 500, shuffle=False
)

torch.manual_seed(0)
device = torch.device('cuda')

model = ConvNet().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.5)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, 3):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)
    
test_samples = enumerate(test_dataloader)
batch_idx, (sample_data, sample_targets) = next(test_samples)
sample_data, sample_targets = sample_data.to(device), sample_targets.to(device)
plt.imshow(sample_data[0][0].cpu(), cmap='gray', interpolation='none')

print(f"Model prediction is : {model(sample_data).data.max(1)[1][0]}")
print(f"Ground Truth is : {sample_targets[0]}")