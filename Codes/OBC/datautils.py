import os
import sys
sys.path.append('yolov5')

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pytorch_lightning as pl
def set_seed(seed):
    # np.random.seed(seed)
    # torch.random.manual_seed(seed)
    pl.seed_everything(seed)

def random_subset(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])

_IMAGENET_RGB_MEANS = (0.5071, 0.4865, 0.4409)
_IMAGENET_RGB_STDS = (0.2673, 0.2564, 0.2762)

def get_imagenet(path, noaug=False, keep_file=''):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])

    train_dataset = datasets.CIFAR100(path,train=True,transform=train_transform)
    if keep_file is not None:        
        keep_bool = np.load(keep_file)
        keep = np.where(keep_bool)[0]
        cali_dataset = torch.utils.data.Subset(train_dataset, keep)
    test_dataset = datasets.CIFAR100(path,train=False,transform=test_transform)

    return train_dataset, cali_dataset, test_dataset

class YOLOv5Wrapper(Dataset):
    def __init__(self, original):
        self.original = original
    def __len__(self):
        return len(self.original)
    def __getitem__(self, idx):
        tmp = list(self.original[idx])
        tmp[0] = tmp[0].float() / 255
        return tmp

def get_coco(path, batchsize):
    from yolov5.utils.datasets import LoadImagesAndLabels
    train_data = LoadImagesAndLabels(
        os.path.join(path, 'images/calib'), batch_size=batchsize
    )
    train_data = YOLOv5Wrapper(train_data)
    train_data.collate_fn = LoadImagesAndLabels.collate_fn
    test_data = LoadImagesAndLabels(
        os.path.join(path, 'images/val2017'), batch_size=batchsize, pad=.5
    )
    test_data = YOLOv5Wrapper(test_data)
    test_data.collate_fn = LoadImagesAndLabels.collate_fn
    return train_data, test_data


DEFAULT_PATHS = {
    'imagenet': [
        '../imagenet'
    ],
    'coco': [
        '../coco'
    ]
}

def get_loaders(
    name, path='', batchsize=-1, workers=8, nsamples=1024, seed=0,
    noaug=False, keep_file=''
):
    if name == 'squad':
        if batchsize == -1:
            batchsize = 16
        import bertsquad
        set_seed(seed)
        return bertsquad.get_dataloader(batchsize, nsamples), None

    if not path:
        for path in DEFAULT_PATHS[name]:
            if os.path.exists(path):
                break

    if name == 'imagenet':
        if batchsize == -1:
            batchsize = 128
        train_data, cali_data, test_data = get_imagenet(path, noaug=noaug, keep_file=keep_file)
        cali_data = random_subset(cali_data, nsamples, seed)
    if name == 'coco':
        if batchsize == -1:
            batchsize = 16
        train_data, test_data = get_coco(path, batchsize)

    collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else None
    trainloader = DataLoader(
        train_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False,
        collate_fn=collate_fn
    )
    collate_fn = train_data.collate_fn if hasattr(cali_data, 'collate_fn') else None
    caliloader = DataLoader(
        cali_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True,
        collate_fn=collate_fn
    )
    collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
    testloader = DataLoader(
        test_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False,
        collate_fn=collate_fn
    )

    return trainloader, caliloader, testloader
