# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import sys; sys.path.append('Utils')
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=False, type=bool)
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--model_dir", default='', type=str)
parser.add_argument("--dataset", default='', type=str)
parser.add_argument("--dpath", default='', type=str)

temp_args, _ = parser.parse_known_args()
if temp_args.config:
    args = parser.parse_args([])
    update_args_from_config(args, config='config.json')
    args = parser.parse_args(namespace=args)
else:
    args = parser.parse_args()

def get_dataset(dataset, root):
    if dataset == 'cifar10':
        train_ds = CIFAR10(root=root, train=True, download=True, transform=cifar10_train_trans)
        return train_ds
    if dataset == 'cifar100':
        train_ds = CIFAR100(root=root, train=True, download=True, transform=cifar10_train_trans)
        return train_ds
    elif dataset == 'tiny-imagenet':  
        train_ds = TinyImageNet(root, train=True, transform=tiny_train_trans)
        return train_ds
    else:
        print(f'{dataset} has not been implemented')
        raise NotImplementedError

@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    train_ds = get_dataset(args.dataset, args.dpath)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    for path in os.listdir(args.model_dir):
        if args.model == "resnet18":
            if args.dataset == 'cifar10':
                m = models.resnet18(weights=None, num_classes=10)
                m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                m.maxpool = nn.Identity()
            elif args.dataset == 'cifar100':
                m = models.resnet18(weights=None, num_classes=100)
                m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                m.maxpool = nn.Identity()
            elif args.dataset == 'tiny-imagenet':
                m = models.resnet18(weights=None, num_classes=200)
                m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                m.maxpool = nn.Identity()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        m.load_state_dict(torch.load(os.path.join(args.model_dir, path, "model.pt")))
        m.to(DEVICE)
        m.eval()

        logits_n = []
        for i in range(args.n_queries):
            logits = []
            for x, _ in tqdm(train_dl):
                x = x.to(DEVICE)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)

        np.save(os.path.join(args.model_dir, path, "logits.npy"), logits_n)


if __name__ == "__main__":
    run()
