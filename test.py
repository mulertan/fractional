import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn.init as init

test_loss, test_acc = [], []


def test(epoch):
    net.eval()
    total = 0
    correct = 0
    t_loss = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        t_loss += loss.item()

        total += targets.size(0)
        targests = targets.view(-1, 1)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.eq(predicted, targets).cpu().sum()

    test_loss.append(t_loss / (batch_idx + 1))
    test_acc.append(correct / total)
    print('test_loss:%.3f,test_acc:%.3f' % (test_loss[-1], test_acc[-1]))