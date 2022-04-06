from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

cudnn.benchmark = True
plt.ion()   # interactive mode

import model1


if __name__ == '__main__':
    print("Test model: ", model1.ALIAS)

    dataset_path = "./Dataset/"
    data_dir_name = input("Enter the name of datset: Ex. data-1: ")
    data_dir = dataset_path + data_dir_name

    dataloaders, dataset_sizes, class_names, device = model1.load_data(data_dir=data_dir)

    print(dataset_sizes, device)

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = model1.model_finetune(1, class_names, device)

    model_ft = model1.train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

    print("\n\nState_dict: *********************\n", model_ft.state_dict().keys())