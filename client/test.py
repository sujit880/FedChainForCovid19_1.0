from __future__ import print_function, division
from client.main import ALIAS

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
import modman

ip_address = "172.16.26.15"  # server macine ip address
# API endpoint
URL = "http://"+ip_address+":5500/api/model/"

P=print
if __name__ == '__main__':

    model1.ALIAS = input("Enter the name of model: Ex. vgg16/resnet18/densenet: ")
    
    print("TModel: ", model1.ALIAS)
    dataset_path = "./Dataset/"
    data_dir_name = input("Enter the name of datset: Ex. data-1: ")
    data_dir = dataset_path + data_dir_name

    dataloaders, dataset_sizes, class_names, device = model1.load_data(data_dir=data_dir)

    print(dataset_sizes, device)
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = model1.model_finetune(model1.ALIAS, class_names, device)
    
    ##############################################
    # Fetch Initial Model Params (If Available)
    ##############################################

    # reply = modman.send_model_params(
    #         URL, modman.convert_tensor_to_list(pie.Q.state_dict()), PIE_PARAMS.LR, ALIAS)
    # print("Response:",reply)

    while modman.get_model_lock(URL, ALIAS):  # wait if model updation is going on
        print("Waiting for Model Lock Release.")
    print("before fetch params")
    global_params, n_push, log_id, is_available = modman.fetch_params(URL, ALIAS)
    # print("global params", global_params)
    n_steps=n_push
    print("After fetch Params")



    if is_available:
        P("Model exist")
        P("Loading Q params .....")
        P("Number Push: ", n_push)
        P("Loading T params .....")
    else:
        P("Setting model for server")
        global_params, n_push, log_id, Iteration = modman.send_model_params(
            URL, modman.convert_tensor_to_list(pie.Q.state_dict()), PIE_PARAMS.LR, ALIAS)
        P("Number Push: ", n_push)
        # print(reply)


    ##############################################
    # Training
    ##############################################
    P('#', 'Train')
    P('Start Training...')
    

    model_ft = model1.train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=n_push)

    print("\n\nState_dict: *********************\n", model_ft.state_dict().keys())