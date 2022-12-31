import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                            transform=transform_)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True)

testset = datasets.CIFAR10(root='~/data', train=True, download=True,
                           transform=transform_)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)