import torch
import torch.nn as nn
import argparse
from torchvision import models
from quantizer import QuantModule
import numpy as np
from adaround import layer_reconstruction
import os

import sys; 
dir = os.path.dirname(__file__)
path = os.path.join(dir, '..', 'Utils')
sys.path.append(path)
from utils import *
DEVICE = select_and_set_device()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config',action='store_true')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--load',type=str,default='')
parser.add_argument('--datapath',type=str,default='')
parser.add_argument('--wbits',type=float,default=32)
parser.add_argument('--save',type=str,default='')
parser.add_argument('--asym',type=bool,default=True)
parser.add_argument('--keep',type=str,default='')
parser.add_argument('--logit_save_path',type=str,default='../dat/')
parser.add_argument('--nqueries',type=int,default=2)
parser.add_argument('--dataset',type=str,default='tiny-imagenet')
parser.add_argument('--last_layer_8bit', action='store_true')

args = parser.parse_args()
if args.config:
    update_args_from_config(args)
    args = parser.parse_args(namespace=args)
    

seed_all(args.seed)

def replace_with_quant_modules(model, weight_quant_params, act_quant_params):
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            quant_module = QuantModule(
                org_module=module,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params
            )
            setattr(model, name, quant_module)
        elif isinstance(module, nn.Sequential):
            replace_with_quant_modules(module, weight_quant_params, act_quant_params)
        elif isinstance(module, nn.Module):
            replace_with_quant_modules(module, weight_quant_params, act_quant_params)
            
def set_last_layer_to_8bit(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, QuantModule):
            module_list += [m]    
    module_list[-1].weight_quantizer.bitwidth_refactor(8)

nclasses = 0
if args.dataset == 'tiny-imagenet':
    nclasses = 200
elif args.dataset == 'cifar10':
    nclasses = 10
elif args.dataset == 'cifar100':
    nclasses = 100
else:
    raise NotImplementedError

model = models.resnet18(weights=None, num_classes=nclasses)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.load_state_dict(torch.load(args.load,weights_only=True,map_location=DEVICE))
model.to(DEVICE)
model.eval()

seed_all(args.seed)
caliloader, testloader = get_dataloaders_with_keepfile(args.dataset, args.datapath, nsamples=-1, batchsize=32, keep_file = args.keep)
trainloader = get_train_loader(args.dataset, dpath=args.datapath, batchsize=32)
true_trainloader = get_train_loader_with_keepfile(args.dataset, args.datapath, 32, args.keep)

print(f'accuracy: {get_acc(model, testloader):.4f}')

# build quantization parameters
wq_params = {'n_bits': args.wbits, 'channel_wise': True, 'scale_method': 'mse', "symmetric": not args.asym}
aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': False}

# replace all conv2d and linear layers with quantized versions
def find_modules(model):
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            quant_module = QuantModule(
                org_module=module,
                weight_quant_params=wq_params,
                act_quant_params=aq_params
            )
            setattr(model, name, quant_module)
        elif isinstance(module, nn.Sequential):
            find_modules(module)
        elif isinstance(module, nn.Module):
            find_modules(module)            
find_modules(model)

if args.last_layer_8bit:
    set_last_layer_to_8bit(model)

# start quantization
for module in model.modules():
    if isinstance(module, QuantModule):
        module.set_quant_state(weight_quant=True, act_quant=False)


print(f'evaluating before AdaRound: {get_acc(model, testloader):.2f}')
# apply adaround
def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]

cali_data = get_train_samples(caliloader, num_samples=1024)

with torch.no_grad():
    for inputs, _ in caliloader:
        inputs = inputs.to(DEVICE)
        _ = model(inputs)

for name, module in model.named_modules():
    if isinstance(module, QuantModule):
        if module.ignore_reconstruction is True:
            print('Ignore reconstruction of layer {}'.format(name))
            continue
        else:
            print('Reconstruction for layer {}'.format(name))
            layer_reconstruction(model, module, cali_data)    

for module in model.modules():
    if isinstance(module, QuantModule):
        module.set_quant_state(weight_quant=True, act_quant=False)

model.eval()
accu = get_acc(model, testloader)
print(f'evaluating: {accu:.2f}')

if args.save:
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    save_train_test_accuracy(model, true_trainloader, testloader, args)

     
def run_inference(queries, train_dl, model, save_dir):
    model.eval()
    logits_n = []
    for i in range(queries):
        logits = []
        for x, _ in train_dl:
            x = x.to(DEVICE)
            outputs = model(x)
            logits.append(outputs.cpu().detach().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)
    
    dir = os.path.join(save_dir, f"AdaRound/W{args.wbits}A32")
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, 'logits.npy')
    np.save(path, logits_n)
    
# run_inference(args.nqueries, trainloader, model, args.logit_save_path)
save_dir = os.path.dirname(args.save)
inference_and_score(args.dataset, args.datapath, save_dir, args.nqueries, trainloader, model)
