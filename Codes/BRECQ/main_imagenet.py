import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import hubconf
from quant import *
from data.imagenet import build_imagenet_data
import pytorch_lightning as pl

import sys; sys.path.append('../Utils')
from utils import *
DEVICE = torch.device(select_and_set_device())

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(cali_loader, num_samples):
    train_data = []
    for batch in cali_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default=True, type=bool)
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--datapath', default='', type=str, help='path to ImageNet data')
    parser.add_argument('--save', default='q_model.pt', type=str)
    parser.add_argument('--load',type=str,default='')

    # quantization parameters
    parser.add_argument('--wbits', default=32, type=float, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--set_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--asym', default=True, type=bool, help='asymmetric rounding')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')
    
    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    
    parser.add_argument('--keep',type=str,default='')
    parser.add_argument('--logit_save_path',type=str,default='../dat/')
    parser.add_argument('--nqueries',type=int,default=2)

    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument('--last_layer_8bit',type=bool,default=False)

    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        args = parser.parse_args([])
        update_args_from_config(args, config='../config.json')
        args = parser.parse_args(namespace=args)
    else:
        args = parser.parse_args()

    seed_all(args.seed)

    # build imagenet data loader
    cali_loader, test_loader = get_dataloaders_with_keepfile(args.dataset, args.datapath, -1, args.batch_size, args.keep)
    train_loader = get_train_loader(args.dataset, args.datapath, args.batch_size)
    true_trainloader = get_train_loader_with_keepfile(args.dataset, args.datapath, args.batch_size, args.keep)

    nclasses = 0
    if args.dataset == 'cifar10': nclasses = 10
    elif args.dataset == 'cifar100': nclasses = 100
    elif args.dataset == 'tiny-imagenet': nclasses = 200
    else: raise NotImplementedError

    # load model
    cnn = hubconf.resnet18(pretrained=False, num_classes=nclasses)
    cnn.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    cnn.maxpool = nn.Identity()
    cnn.to(DEVICE)
    cnn.load_state_dict(torch.load(args.load, weights_only=True))
    cnn.eval()
    # build quantization parameters
    wq_params = {'n_bits': args.wbits, 'channel_wise': args.channel_wise, 'scale_method': 'mse', "symmetric": not args.asym}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params, ignore_last=False)
    qnn.to(DEVICE)
    qnn.eval()
    
    if args.set_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
        
    if args.last_layer_8bit:
        module_list = []
        for m in qnn.modules():
            if isinstance(m, QuantModule):
                module_list += [m]    
        module_list[-1].weight_quantizer.bitwidth_refactor(8)        

    cali_data = get_train_samples(cali_loader, num_samples=args.num_samples)

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)    
    
    _ = qnn(cali_data[:64].to(DEVICE))
    print('Quantized accuracy before brecq: {}'.format(get_acc(qnn, test_loader)))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start calibration
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    accu = get_acc(qnn, test_loader)
    print('Weight quantization accuracy: {}'.format(accu))

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(DEVICE))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.wbits, args.n_bits_a,
                                                               validate_model(test_loader, qnn)))
     
    if args.save is not None:
        torch.save(qnn.state_dict(), args.save)
        save_train_test_accuracy(qnn, true_trainloader, test_loader, args)
        
            
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
        
        dir = os.path.join(save_dir, f"BRECQ/W{args.wbits}A32")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, f'logits.npy')
        np.save(path, logits_n)
        
    run_inference(args.nqueries, train_loader, qnn, args.logit_save_path)