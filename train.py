'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
best_acc = {'net_1': 0, 'net_2': 0}  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train_1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset_1 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train_1)
trainloader_1 = torch.utils.data.DataLoader(
    trainset_1, batch_size=128, shuffle=True, num_workers=2)

trainset_2 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train_2)
trainloader_2 = torch.utils.data.DataLoader(
    trainset_2, batch_size=128, shuffle=True, num_workers=2)

testset_1 = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test_1)
testloader_1 = torch.utils.data.DataLoader(
    testset_1, batch_size=100, shuffle=False, num_workers=2)

testset_2 = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test_2)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net_1 = ResNet18()
net_2 = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net_1 = net_1.to(device)
net_2 = net_2.to(device)
if device == 'cuda':
    net_1 = torch.nn.DataParallel(net_1)
    net_2 = torch.nn.DataParallel(net_2)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net_1.load_state_dict(checkpoint['net'])
    net_2.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer_1 = optim.SGD(net_1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_2 = optim.SGD(net_2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=200)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=200)


# Training
def train(model, optimizer, trainloader, epoch, pre):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print(pre)
        progress_bar(batch_idx, len(trainloader), '%s Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (pre, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), '%s Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (pre, test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(model, testloader, epoch, pre):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print(pre)
            progress_bar(batch_idx, len(testloader), '%s Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (pre, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc[pre]:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'{pre}'):
            os.mkdir(f'{pre}')
        torch.save(state, f'./{pre}/ckpt.pth')

        best_acc[pre] = acc
        print(f'{pre} best acc: {best_acc[pre]}')


for epoch in range(start_epoch, start_epoch+200):
    train(net_1, optimizer_1, trainloader_1, epoch, 'net_1')
    test(net_1, testloader_1, epoch, 'net_1')

    train(net_2, optimizer_2, trainloader_2, epoch, 'net_2')
    test(net_2, testloader_2, epoch, 'net_2')

    scheduler_1.step()
    scheduler_2.step()
