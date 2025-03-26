import argparse
import subprocess
import random
import os

work_dir = os.path.abspath(os.path.dirname(__file__))
quant_methods = {'AdaRound':'main_quant.py', 'BRECQ':'main_imagenet.py', 'OBC':'main_trueobs.py'}
bit_str = {1:'1', 1.58:'log3', 2:'2', 3:'3', 4:'4'}


def shadow_models_trainer():
    for shadow_id in range(args.n_shadows):
        subprocess.run(
            [
                'python', 
                os.path.join(work_dir, 'train.py'), 
                '--dataset', args.dataset, 
                '--datapath', datapath, 
                '--seed', f'{random.randint()}',
                '--n_shadows', f'{args.n_shadows}',
                '--shadow_id', f'{shadow_id}',
                '--savedir', os.path.join(work_dir, 'shadows', f'{args.dataset}', f'{shadow_id}')
            ],check=True
        )

def target_model_trainer():
    subprocess.run(
        [
            'python', 
            os.path.join(work_dir, 'train.py'), 
            '--dataset', args.dataset, 
            '--datapath', datapath, 
            '--seed', f'{args.seed}',
            '--savedir', os.path.join(work_dir, 'models', f'{args.dataset}')
        ],check=True
    )
    
def quantizer():
    for method in quant_methods.keys():    
        for bit in args.quant_range: 
            subprocess.run(
                [
                    'python',
                    os.path.join(work_dir, method, quant_methods[method]),
                    '--dataset', args.dataset,
                    '--datapath', datapath,
                    '--seed', f'{args.seed}',
                    '--load', os.path.join(work_dir, 'models', f'{args.dataset}', 'model.pt'),
                    '--keep', os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                    '--wbits', f'{bit}',
                    '--save', os.path.join(work_dir, 'quant_models', f'{args.dataset}', method, f'w{bit_str[bit]}_quant','model.pt')
                ],check=True
            )
            
def plot():
    for legend in ['full', 'AdaRound', 'BRECQ', 'OBC']:
        if legend == 'full':
            subprocess.run(
                [
                    'python',
                    'plot.py',
                    '--model_dir',os.path.join(work_dir, 'shadows', f'{args.dataset}'),
                    '--keep',os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                    '--scores',os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                    '--name','Full Prec'
                ],check=True
            )
        else:
            parent = os.path.join(work_dir, 'quant_models', f'{args.dataset}', legend)
            dir_list = os.listdir(parent)
            for dirname in dir_list:
                subprocess.run(
                    [
                        'python',
                        'plot.py',
                        '--model_dir',os.path.join(work_dir, 'shadows', f'{args.dataset}'),
                        '--keep',os.path.join(dirname, 'keep.npy'),
                        '--scores',os.path.join(dirname, 'keep.npy'),
                        '--name','Full Prec'
                    ],check=True
                )
            
parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset', choices=['cifar10', 'cifar100', 'tiny-imagenet'], help='Dataset to use', type=str)
parser.add_argument(
    'datapath', help='Path to the dataset', type=str)
parser.add_argument(
    '--n_shadows', help='Number of shadows models to create', type=int, default=64)
parser.add_argument(
    '--quant_range', help='Quantization range', type=float, nargs='+', default=[1.58, 2, 3, 4])
parser.add_argument(
    '--seed', help='Seed for reproducibility', type=int, default=42)

args = parser.parse_args()    
datapath = os.path.abspath(args.datapath)
    
# target_model_trainer()
# shadow_models_trainer()
quantizer()
plot()
