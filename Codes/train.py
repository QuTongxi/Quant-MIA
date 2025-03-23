# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


upath = os.path.abspath(os.path.join(os.path.dirname(__file__), './Utils'))
import sys; sys.path.append(upath)
from utils import *

DEVICE = select_and_set_device()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=True, type=bool)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--n_shadows", default=None, type=int)
parser.add_argument("--shadow_id", default=0, type=int)
parser.add_argument("--model", default="resnet18", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--dataset", default='', type=str)
parser.add_argument("--datapath", default='', type=str)

temp_args, _ = parser.parse_known_args()
if temp_args.config:
    args = parser.parse_args([])
    update_args_from_config(args, config='config.json')
    args = parser.parse_args(namespace=args)
else:
    args = parser.parse_args()

def run():
    pl.seed_everything(args.seed)

    # Dataset
    nclasses = 0
    seed_all(args.seed)
    if args.dataset == 'cifar10':
        nclasses = 10
        train_ds = CIFAR10(args.datapath, train=True, transform=cifar10_train_trans, download=True)
        test_ds = CIFAR10(args.datapath, train=False, transform=cifar10_test_trans, download=True) 
    elif args.dataset == 'cifar100':
        nclasses = 100
        train_ds = CIFAR100(args.datapath, train=True, transform=cifar100_train_trans, download=True)
        test_ds = CIFAR100(args.datapath, train=False, transform=cifar100_test_trans, download=True)
    elif args.dataset == 'tiny-imagenet':
        nclasses = 200
        train_ds = TinyImageNet(args.datapath, train=True, transform=tiny_train_trans)
        test_ds = TinyImageNet(args.datapath, train=False, transform=tiny_test_trans)
    else:
        raise NotImplementedError

    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)

    np.random.seed(args.seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model
    m = models.resnet18(weights=None, num_classes=nclasses)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

    m.eval()
    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")
    
    save_target_train_test_accuracy(m, train_dl, test_dl, args)

if __name__ == "__main__":
    run()
