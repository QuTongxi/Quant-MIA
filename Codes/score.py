# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified copy by Chenxiang Zhang (orientino) of the original:
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

import sys; sys.path.append('Utils')
from utils import *
available_device = select_best_gpu()
if available_device.startswith("cuda"):
    os.environ['CUDA_VISIBLE_DEVICES'] = available_device.split(":")[1]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=False, type=bool)
parser.add_argument("--model_dir", default='./models/cifar100/', type=str)
parser.add_argument("--dpath", default='../Datasets/cifar100/', type=str)
parser.add_argument("--dataset", default='cifar100', type=str)

temp_args, _ = parser.parse_known_args()
if temp_args.config:
    args = parser.parse_args([])
    update_args_from_config(args, config='config.json')
    args = parser.parse_args(namespace=args)
else:
    args = parser.parse_args()

def load_one(path):
    """
    This loads a logits and converts it to a scored prediction.
    """
    opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]

    # Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = get_labels()  # TODO generalize this

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path, "scores.npy"), logit)

def get_labels():
    if args.dataset == 'cifar10':
        dataset = CIFAR10(root=args.dpath, train=True, download=False)
        return np.array(dataset.targets)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100(root=args.dpath, train=True, download=False)
        return np.array(dataset.targets)
    elif args.dataset == 'tiny-imagenet':
        dataset = TinyImageNet(args.dpath, train=True, transform=None)
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label)
        return np.array(all_labels)
    else:
        raise NotImplementedError

def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [os.path.join(args.model_dir, x) for x in os.listdir(args.model_dir)])

if __name__ == "__main__":
    load_stats()
