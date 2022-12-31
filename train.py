from models import ResNet
import dataset
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn.init as init


net = ResNet18().to(device)
criterion =nn.CrossEntropyLoss()
frac_criterion = nn.CosineSimilarity()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                      weight_decay =decay)

train_acc, train_loss = [], []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    tr_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        alpha = 0.5
        labels = generate_label(targets, alpha).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = torch.mean(- frac_criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        targests = targets.view(-1, 1)
        correct += torch.eq(predicted, targets).cpu().sum()

    train_loss.append(tr_loss / (batch_idx + 1))
    train_acc.append(correct / total)
    print('train_loss:%.3f,train_acc:%.3f' % (train_loss[-1], train_acc[-1]))


for epoch in range(epochs):
    if epoch == 70:
        lr /= 10

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train(epoch)