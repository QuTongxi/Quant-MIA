import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
import numpy as np

def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False, keep_file: str = ''):
    print('==> Using Pytorch Dataset')
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
        ]
    )

    datadir = data_path
    train_dataset = CIFAR100(root=datadir, train=True, download=True, transform=train_transform)
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
        
    val_dataset = CIFAR100(root=datadir, train=False, download=True, transform=test_transform)

    if dist_sample:
        cali_sampler = torch.utils.data.distributed.DistributedSampler(cali_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        cali_sampler = None
        val_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    cali_loader = torch.utils.data.DataLoader(
        cali_dataset, batch_size=batch_size, shuffle=(cali_sampler is None),
        num_workers=workers, pin_memory=True, sampler=cali_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, cali_loader, val_loader
