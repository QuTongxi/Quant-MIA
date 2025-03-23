import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
import pytorch_lightning as pl
import time

from tiny_imagenet import *

tiny_train_trans = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ]
)
tiny_test_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ]
)
cifar10_train_trans = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ]
)
cifar10_test_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ]
)
cifar100_train_trans = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    ]
)
cifar100_test_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    ]
)

my_seed = None
selected_device = None

def seed_all(seed):
    global my_seed
    my_seed = seed
    
def seed_here():
    assert my_seed is not None, "use seed_all before loading datasets"
    pl.seed_everything(my_seed)
    np.random.seed(my_seed)    

def random_subset(data, nsamples):
    seed_here()
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])

def _cali_test_loaders(train_dataset, test_dataset, nsamples:int, batchsize:int, keep_file:str):
    test_loader = DataLoader(test_dataset,batch_size=batchsize, shuffle=False,num_workers=4, pin_memory=True) 
    if keep_file is not None:     
        print('keep file is not none.')   
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
        if nsamples != -1:
            cali_dataset = random_subset(cali_dataset, nsamples)
            
        cali_loader = DataLoader(cali_dataset, batch_size=batchsize, shuffle=True,num_workers=4, pin_memory=True)
        return cali_loader, test_loader         
    else:
        print('no keep file found')
        if nsamples != -1:
            train_dataset = random_subset(train_dataset, nsamples)
            train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,num_workers=4, pin_memory=True)
        return train_loader, test_loader


def get_cifar10_loaders_with_keepfile(dpath:str, nsamples:int, batchsize:int, keep_file:str):
    train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=cifar10_train_trans)
    test_dataset = CIFAR10(root=dpath, train=False, download=True, transform=cifar10_test_trans)
    return _cali_test_loaders(train_dataset, test_dataset, nsamples, batchsize, keep_file)

def get_cifar100_loaders_with_keepfile(dpath:str, nsamples:int, batchsize:int, keep_file:str):
    train_dataset = CIFAR100(root=dpath, train=True, download=True, transform=cifar100_train_trans)
    test_dataset = CIFAR100(root=dpath, train=False, download=True, transform=cifar100_test_trans)
    return _cali_test_loaders(train_dataset, test_dataset, nsamples, batchsize, keep_file)

def get_tiny_imagenet_loaders_with_keepfile(dpath:str, nsamples:int, batchsize:int, keep_file:str):
    train_dataset = TinyImageNet(dpath, train=True, transform=tiny_train_trans)
    test_dataset = TinyImageNet(dpath, train=False, transform=tiny_test_trans)
    return _cali_test_loaders(train_dataset, test_dataset, nsamples, batchsize, keep_file)

def get_dataloaders_with_keepfile(dataset:str, dpath:str, nsamples:int=-1, batchsize:int=32, keep_file:str=None):
    seed_here()
    if dataset == 'cifar10':
        return get_cifar10_loaders_with_keepfile(dpath=dpath,nsamples=nsamples,batchsize=batchsize,keep_file=keep_file)
    elif dataset == 'tiny-imagenet':
        return get_tiny_imagenet_loaders_with_keepfile(dpath=dpath,nsamples=nsamples,batchsize=batchsize,keep_file=keep_file)
    elif dataset == 'cifar100':
        return get_cifar100_loaders_with_keepfile(dpath=dpath,nsamples=nsamples,batchsize=batchsize,keep_file=keep_file)
    else:
        raise NotImplementedError

def get_train_loader(dataset:str, dpath:str, batchsize:int, shuffle:bool=False):
    seed_here()
    if dataset == 'cifar10':
        train_set = CIFAR10(root=dpath, train=True,transform=cifar10_train_trans)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=shuffle, num_workers=4, pin_memory=True)
        return train_loader
    elif dataset == 'cifar100':
        train_set = CIFAR100(root=dpath, train=True,transform=cifar100_train_trans)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=shuffle, num_workers=4, pin_memory=True)
        return train_loader        
    elif dataset == 'tiny-imagenet':
        train_set = TinyImageNet(dpath, train=True, transform=tiny_train_trans)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=shuffle,num_workers=4, pin_memory=True)
        return train_loader
    else:
        raise NotImplementedError    
    
def get_train_loader_with_keepfile(dataset, dpath, batchsize, keep_file):
    seed_here()
    trainloader, _ = get_dataloaders_with_keepfile(dataset, dpath, nsamples=-1, batchsize=batchsize, keep_file=keep_file)    
    return trainloader

def select_best_gpu():
    import pynvml
    pynvml.nvmlInit()  
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count == 0:
        device = "cpu"
    else:
        gpu_id, max_free_mem = 0, 0.
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_free = round(pynvml.nvmlDeviceGetMemoryInfo(handle).free/(1024*1024*1024), 3)
            if memory_free > max_free_mem:
                gpu_id = i
                max_free_mem = memory_free
        device = f"cuda:{gpu_id}"
        print(f"total have {gpu_count} gpus, max gpu free memory is {max_free_mem}, which gpu id is {gpu_id}")
    return device


def select_and_set_device(dev_id = -1):
    global selected_device
    if selected_device is None:   
        if dev_id != -1:
            selected_device  = f"cuda:{dev_id}"
        else:    
            selected_device  = select_best_gpu()    
    return selected_device
    


def update_args_from_config(args, config='../config.json'):
    import inspect
    import json
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    caller_filename = os.path.basename(module.__file__)
    
    with open(config, 'r') as f:
        config_args = json.load(f)
        print('load from config file')
    
    if caller_filename in config_args:
        caller_config = config_args[caller_filename]
        for key, value in caller_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

@torch.no_grad()
def get_acc(model, test_loader):
    model.eval()
    acc = []
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)
    return acc.item() * 100

fname_dict = {
    'main_quant.py': 'AdaRound',
    'main_trueobs.py': 'OBC',
    'train.py': 'Full',
    'main_imagenet.py': 'BRECQ'
}


def save_train_test_accuracy(model, true_trainloader, testloader, args):
    train_acc = get_acc(model, true_trainloader)
    test_acc = get_acc(model, testloader)
    
    import inspect
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    fname = os.path.basename(module.__file__)
    
    content = f"""[{fname_dict[fname]}] dataset: {args.dataset} wbits: {args.wbits}
Train Accu.: {train_acc:.4f} Test Accu.: {test_acc:.4f}
"""
    
    print(content)
    filename = os.path.join(os.path.dirname(__file__), '..', 'output.txt')
    with open(filename, 'a') as f:
        f.write(content)   

def save_target_train_test_accuracy(model, true_trainloader, testloader, args):
    train_acc = get_acc(model, true_trainloader)
    test_acc = get_acc(model, testloader)
    
    import inspect
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    fname = os.path.basename(module.__file__)
    
    content = f"""[{fname_dict[fname]}] dataset: {args.dataset} epochs: {args.epochs} pkeep: {args.pkeep}
Train Accu.: {train_acc:.4f} Test Accu.: {test_acc:.4f}
"""
    
    print(content)
    filename = os.path.join(os.path.dirname(__file__), '..', 'output.txt')
    with open(filename, 'a') as f:
        f.write(content) 

class ModelAnalyzer:
    def __init__(self, model):
        """
        Initializes the ModelAnalyzer with a given model.
        
        :param model: The PyTorch model to be analyzed.
        """
        self.model = model

    def write_layer_info(self, layer, file_handler, level=0):
        indent = ' ' * (level * 4)  # Indentation for hierarchy
        
        for name, param in layer.named_parameters():
            if param is not None:
                param_num = param.numel()
                dtype = param.dtype
                max_val = param.max().item() if param.numel() > 0 else 'N/A'
                min_val = param.min().item() if param.numel() > 0 else 'N/A'
                
                file_handler.write(f"{indent}{name}:\n")
                file_handler.write(f"{indent}  Number of parameters: {param_num}\n")
                file_handler.write(f"{indent}  Data type: {dtype}\n")
                file_handler.write(f"{indent}  Maximum value: {max_val}\n")
                file_handler.write(f"{indent}  Minimum value: {min_val}\n")

    def traverse_model(self, model, file_handler, level=0):
        """
        Recursively traverses through all layers of the provided model and writes details.
        
        :param model: The current model or sub-model to traverse.
        :param file_handler: The file handler where the analysis will be written.
        :param level: Current depth level in the model's hierarchy.
        """
        for name, child in model.named_children():
            file_handler.write(f"{' ' * (level * 4)}{name}:\n")
            if hasattr(child, 'named_parameters'):
                self.write_layer_info(child, file_handler, level + 1)
            self.traverse_model(child, file_handler, level + 1)


    def analyze_model(self, out_file):
        """
        Analyzes the model and writes the hierarchy and parameter details to the specified file.
        
        :param out_file: The path to the output file.
        """
        with open(out_file, 'w') as fh:
            print("Model Hierarchy and Parameter Details:\n")
            self.traverse_model(self.model, fh)
            print("\nAnalysis Completed.\n")

    def measure_model_time(self, test_loader, out_file, legend=''):
        """
        Measures the total and per-sample inference time of the model on the provided data loader.
        
        :param test_loader: DataLoader for the test dataset.
        :param out_file: The path to the output file where results will be appended.
        """
        self.model.eval()
        self.model.cuda()
        total_time = 0

        start_time = time.time()

        with torch.no_grad():
            for data, _ in test_loader:
                batch_start_time = time.time()
                data = data.cuda()
                output = self.model(data)
                total_time += time.time() - batch_start_time

        end_time = time.time()
        overall_time = end_time - start_time
        sample_count = len(test_loader.dataset)
        avg_sample_time = total_time / sample_count

        with open(out_file, 'a') as file:
            file.write(f"Test Begin.{legend}\n")
            file.write(f"Total time taken: {overall_time:.4f} seconds\n")
            file.write(f"Average time per sample: {avg_sample_time:.4f} seconds\n")
            file.write(f"Number of samples processed: {sample_count}\n")

        print("Measurement complete. Results have been written to", out_file)
            


         