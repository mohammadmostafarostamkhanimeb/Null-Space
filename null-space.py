import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net_1 = ResNet18()
net_2 = ResNet18()

checkpoint_1 = torch.load(f'./net_1/ckpt.pth')
checkpoint_2 = torch.load(f'./net_2/ckpt.pth')

net_1.load_state_dict(checkpoint_1['net'])
net_2.load_state_dict(checkpoint_2['net'])

acc_1 = checkpoint_1['acc']
epoch_1 = checkpoint_1['epoch']
acc_2 = checkpoint_2['acc']
epoch_2 = checkpoint_2['epoch']

net_1.to(device)
net_2.to(device)

net_1.eval()
net_2.eval()




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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Function to calculate the null space of a matrix
def null_space(A, tol=1e-5):
    U, S, Vh = torch.linalg.svd(A)
    null_space = Vh.T[:, S < tol]
    return null_space

# Function to calculate the row space of a matrix
def row_space(A):
    Q, R = torch.linalg.qr(A)
    rank = torch.linalg.matrix_rank(A)
    row_space_basis = R[:rank]
    return row_space_basis

# Function to calculate the column space of a matrix
def column_space(A):
    A_T = A.T
    Q_T, R_T = torch.linalg.qr(A_T)
    rank_T = torch.linalg.matrix_rank(A_T)
    column_space_basis = R_T[:rank_T].T
    return column_space_basis

# Function to calculate the projection matrix
def projection_matrix(B):
    B_t = B.T
    return B @ torch.linalg.inv(B_t @ B) @ B_t



def NuSA(x, W):
    nusa = torch.sqrt((torch.norm(x)**2)-(torch.norm(torch.matmul(x, null_space(W.T))))**2)/torch.norm(x)
    return nusa


def hook_fn(module, input, output):
    setattr(module, 'input_tensor', input[0])


test_nusa = dict()

net_1.linear.register_forward_hook(hook_fn)
net_2.linear.register_forward_hook(hook_fn)

# for name, module in net_1.named_modules():
#     if isinstance(module, nn.Linear):
#         module.register_forward_hook(hook_fn)

# for name, module in net_2.named_modules():
#     if isinstance(module, nn.Linear):
#         module.register_forward_hook(hook_fn)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Function to calculate the null space of a matrix
def null_space(A, tol=1e-5):
    U, S, Vh = torch.linalg.svd(A)
    null_space = Vh.T[:, S < tol]
    return null_space

# Function to calculate the row space of a matrix
def row_space(A):
    Q, R = torch.linalg.qr(A)
    rank = torch.linalg.matrix_rank(A)
    row_space_basis = R[:rank]
    return row_space_basis

# Function to calculate the column space of a matrix
def column_space(A):
    A_T = A.T
    Q_T, R_T = torch.linalg.qr(A_T)
    rank_T = torch.linalg.matrix_rank(A_T)
    column_space_basis = R_T[:rank_T].T
    return column_space_basis

# Function to calculate the projection matrix
def projection_matrix(B):
    B_t = B.T
    return B @ torch.linalg.inv(B_t @ B) @ B_t



def NuSA(x, W):
    nusa = torch.sqrt(torch.norm((torch.norm(x)**2)-((torch.norm(projection_matrix(W.T) @ x))**2)))/torch.norm(x)
    return nusa
