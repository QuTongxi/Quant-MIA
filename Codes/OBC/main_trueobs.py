import argparse
import copy
import os

import torch
import torch.nn as nn
import torchvision.models as models

from datautils import *
from modelutils import *
from quant import *
from trueobs import *

import sys
dir = os.path.dirname(__file__)
path = os.path.join(dir, '..', 'Utils')
sys.path.append(path)
from utils import *
DEVICE = select_and_set_device()

parser = argparse.ArgumentParser()

parser.add_argument('--config', action='store_true')
parser.add_argument('--model', type=str, default='rn18')
parser.add_argument(
    '--compress', type=str, choices=['quant', 'nmprune', 'unstr', 'struct', 'blocked'], default='quant'
)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--nsamples', type=int, default=2048)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=float, default=32)
parser.add_argument('--abits', type=int, default=32)
parser.add_argument('--wperweight', action='store_true')
parser.add_argument('--asym', type=bool, default=True)
parser.add_argument('--wminmax', action='store_true')
parser.add_argument('--actsym', action='store_true')
parser.add_argument('--aminmax', action='store_true')
parser.add_argument('--rel-damp', type=float, default=0)

parser.add_argument('--prunen', type=int, default=2)
parser.add_argument('--prunem', type=int, default=4)
parser.add_argument('--blocked_size', type=int, default=4)
parser.add_argument('--min-sparsity', type=float, default=0)
parser.add_argument('--max-sparsity', type=float, default=0)
parser.add_argument('--delta-sparse', type=float, default=0)
parser.add_argument('--sparse-dir', type=str, default='')

parser.add_argument('--keep',type=str,default='')
parser.add_argument('--logit_save_path',type=str,default='../dat/')
parser.add_argument('--nqueries',type=int,default=2)

parser.add_argument('--dataset',type=str,default='tiny-imagenet')
parser.add_argument('--last_layer_8bit', action='store_true')
parser.add_argument('--bnt-batches', type=int, default=100)
   
args = parser.parse_args()
if args.config:
    update_args_from_config(args)
    args = parser.parse_args(namespace=args)

seed_all(args.seed)
dataloader, testloader = get_dataloaders_with_keepfile(args.dataset, args.datapath, args.nsamples, 128, args.keep)
true_trainloader = get_train_loader_with_keepfile(args.dataset, args.datapath, 32, args.keep)
trainloader = get_train_loader(args.dataset, args.datapath, 32)

nclasses = 0
if args.dataset == 'tiny-imagenet':
    nclasses = 200
elif args.dataset == 'cifar10':
    nclasses = 10
elif args.dataset == 'cifar100':
    nclasses = 100
else:
    raise NotImplementedError

if args.nrounds == -1:
    args.nrounds = 1 if 'yolo' in args.model or 'bert' in args.model else 10 
    if args.noaug:
        args.nrounds = 1
_, _, run = get_functions(args.model)

def get_model(dataset, load):
    nclasses = 10 if dataset == 'cifar10' else 100 if dataset == 'cifar100' else 200
    model = resnet18(weights=None, num_classes=nclasses)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(DEVICE)
    model.load_state_dict(torch.load(load,weights_only=True,map_location=DEVICE),strict=False)   
    model.eval()
    return model

aquant = args.compress == 'quant' and args.abits < 32
wquant = args.compress == 'quant' and args.wbits < 32

modelp = get_model(args.dataset, args.load)
if aquant:
    add_actquant(modelp)
modeld = get_model(args.dataset, args.load)

layersp = find_layers(modelp)
layersd = find_layers(modeld)

SPARSE_DEFAULTS = {
    'unstr': (0, .99, .1),
    'struct': (0, .9, .05),
    'blocked': (0, .95, .1)
}
sparse = args.compress in SPARSE_DEFAULTS
if sparse: 
    if args.min_sparsity == 0 and args.max_sparsity == 0: 
        defaults = SPARSE_DEFAULTS[args.compress]
        args.min_sparsity, args.max_sparsity, args.delta_sparse = defaults 
    sparsities = []
    density = 1 - args.min_sparsity
    while density > 1 - args.max_sparsity:
        sparsities.append(1 - density)
        density *= 1 - args.delta_sparse
    sparsities.append(args.max_sparsity)
    sds = {s: copy.deepcopy(modelp).cpu().state_dict() for s in sparsities}

trueobs = {}
for name in layersp:
    layer = layersp[name]
    if isinstance(layer, ActQuantWrapper):
        layer = layer.module
    trueobs[name] = TrueOBS(layer, rel_damp=args.rel_damp)
    if aquant:
        layersp[name].quantizer.configure(
            args.abits, sym=args.actsym, mse=not args.aminmax
        )
    if wquant:
        if 'fc' in name and args.last_layer_8bit:
            print('Setting the last layer to 8-bit')
            trueobs[name].quantizer = Quantizer()
            trueobs[name].quantizer.configure(
                8, perchannel=not args.wperweight, sym=not args.asym, mse=not args.wminmax
            )
        else:
            trueobs[name].quantizer = Quantizer()
            trueobs[name].quantizer.configure(
                args.wbits, perchannel=not args.wperweight, sym=not args.asym, mse=not args.wminmax
            )

if not (args.compress == 'quant' and not wquant):
    cache = {}
    def add_batch(name):
        def tmp(layer, inp, out):
            trueobs[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in trueobs:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(args.nrounds):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()
    for name in trueobs:
        print(name)
        if args.compress == 'quant':
            print('Quantizing ...')
            trueobs[name].quantize()
        if args.compress == 'nmprune':
            if trueobs[name].columns % args.prunem == 0:
                print('N:M pruning ...')
                trueobs[name].nmprune(args.prunen, args.prunem)
        if sparse:
            Ws = None
            if args.compress == 'unstr':
                print('Unstructured pruning ...')
                trueobs[name].prepare_unstr()
                Ws = trueobs[name].prune_unstr(sparsities)
            if args.compress == 'struct':
                if not isinstance(trueobs[name].layer, nn.Conv2d):
                    size = 1
                else:
                    tmp = trueobs[name].layer.kernel_size
                    size = tmp[0] * tmp[1]
                if trueobs[name].columns / size > 3:
                    print('Structured pruning ...')
                    Ws = trueobs[name].prune_struct(sparsities, size=size)
            if args.compress == 'blocked':
                if trueobs[name].columns % args.blocked_size == 0:
                    print('Blocked pruning ...')
                    trueobs[name].prepare_blocked(args.blocked_size)
                    Ws = trueobs[name].prune_blocked(sparsities)
            if Ws:
                for sparsity, W in zip(sparsities, Ws):
                    sds[sparsity][name + '.weight'] = W.reshape(sds[sparsity][name + '.weight'].shape).cpu()
        trueobs[name].free()

if sparse:
    if args.sparse_dir:
        for sparsity in sparsities:
            name = '%s_%04d.pth' % (args.model, int(sparsity * 10000))
            torch.save(sds[sparsity], os.path.join(args.sparse_dir, name))
    exit()

if aquant:
    print('Quantizing activations ...')
    def init_actquant(name):
        def tmp(layer, inp, out):
            layersp[name].quantizer.find_params(inp[0].data)
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(init_actquant(name)))
    with torch.no_grad():
        run(modeld, next(iter(dataloader)))
    for h in handles:
        h.remove()
  
print('Evaluating before bnt...', end=' ')
print(f'{get_acc(modelp, testloader):.2f}')

dirpath = os.path.dirname(args.save)
if not os.path.exists(dirpath):
    os.makedirs(dirpath,exist_ok=True)
torch.save(modelp.state_dict(), args.save)

def batchnorm_tuning():
    print('Batchnorm tuning ...')
    modelp = get_model(args.dataset, args.save)
    dataloader,_ = get_dataloaders_with_keepfile(args.dataset, args.datapath, 1024, 128, None)

    loss = 0
    for batch in dataloader:
        loss += run(modelp, batch, loss=True)
    print(loss / args.nsamples)

    batchnorms = find_layers(modelp, [nn.BatchNorm2d])
    for bn in batchnorms.values():
        bn.reset_running_stats()
        bn.momentum = .1
    modelp.train()
    with torch.no_grad():
        i = 0
        while i < args.bnt_batches:
            for batch in dataloader:
                if i == args.bnt_batches:
                    break
                run(modelp, batch)
                i += 1
    modelp.eval()

    loss = 0
    for batch in dataloader:
        loss += run(modelp, batch, loss=True)
    print(loss / args.nsamples)
    torch.save(modelp.state_dict(), args.save)

 
batchnorm_tuning()
 
def run_inference(queries, train_dl, model, save_dir):
    model.eval()
    logits_n = []
    for i in range(queries):
        logits = []
        for x, _ in train_dl:
            x = x.cuda()
            outputs = model(x)
            logits.append(outputs.cpu().detach().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)
    
    name, _ = os.path.splitext(os.path.basename(args.load))
    dir = os.path.join(save_dir, f"OBC/{name}")
    os.makedirs(dir, exist_ok=True) 
    path = os.path.join(dir, f'logits.npy')
    np.save(path, logits_n)

modelp = get_model(args.dataset, args.save)    
print('Evaluating ...', end=' ')
print(f'{get_acc(modelp, testloader):.2f}')
save_train_test_accuracy(modelp, true_trainloader, testloader, args)
# run_inference(args.nqueries, trainloader, modelp, args.logit_save_path)
save_dir = os.path.dirname(args.save)
inference_and_score(args.dataset, args.datapath, save_dir, args.nqueries, trainloader, modelp)

