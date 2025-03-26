import argparse
import subprocess
import random
import os

work_dir = os.path.abspath(os.path.dirname(__file__))
quant_methods = {'AdaRound':'main_quant.py', 'BRECQ':'main_imagenet.py', 'OBC':'main_trueobs.py'}
bit_str = {1:'1', 1.58:'log3', 2:'2', 3:'3', 4:'4'}
datapath = ''
seed = 0

def shadow_models_trainer():
    for shadow_id in range(args.n_shadows):
        subprocess.run(
            [
                'python', 
                os.path.join(work_dir, 'train.py'), 
                '--dataset', args.dataset, 
                '--datapath', datapath, 
                '--seed', seed,
                '--n_shadows', f'{args.n_shadows}',
                '--shadow_id', f'{shadow_id}',
                '--savedir', os.path.join(work_dir, 'shadows', f'{args.dataset}', f'{shadow_id}')
            ],check=True
        )

def target_model_trainer(pkeep = 0.5):
    subprocess.run(
        [
            'python', 
            os.path.join(work_dir, 'train.py'), 
            '--dataset', args.dataset, 
            '--datapath', datapath, 
            '--seed', seed,
            '--savedir', os.path.join(work_dir, 'models', f'{args.dataset}'),
            '--pkeep', f'{pkeep}'
        ],check=True
    )
    
def quantizer():
    for method in quant_methods.keys():    
        for bit in args.quant_range: 
            subprocess.run(
                [
                    'python',
                    os.path.join(work_dir, method, quant_methods[method]),
                    '--config', args.use_config,
                    '--dataset', args.dataset,
                    '--datapath', datapath,
                    '--seed', seed,
                    '--load', os.path.join(work_dir, 'models', f'{args.dataset}', 'model.pt'),
                    '--keep', os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                    '--wbits', f'{bit}',
                    '--save', os.path.join(work_dir, 'quant_models', f'{args.dataset}', method, f'w{bit_str[bit]}_quant','model.pt'),
                    '--asym', not args.sym,
                    '--last_layer_8bit' if args.dataset == 'tiny-imagenet' and args.auto_last_8bit else ''
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
                    '--scores',os.path.join(work_dir, 'models', f'{args.dataset}', 'scores.npy'),
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
                        '--keep',os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                        '--scores',os.path.join(parent ,dirname, 'scores.npy'),
                        '--name', f'{legend} w{dirname[1:]}'
                    ],check=True
                )
                
def Quant_MIA():
    target_model_trainer()
    shadow_models_trainer()
    quantizer()
    plot()    
    
def test_pkeep():
    for pkeep in [0.1, 0.25, 0.5, 0.75, 1.0]:
        target_model_trainer(pkeep)
        quantizer()


if __name__ == '__main__':            
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
        '--seed', help='use a random number if don`t set the seed', type=int, default=None)
    parser.add_argument(
        '--sym', help='Symmetric quantization', action='store_true')
    parser.add_argument(
        '--use_config', help='Use config file', action='store_true')
    parser.add_argument(
        '--cycle_times', help='Cycle times', type=int, default=1)

    args = parser.parse_args()    
    datapath = os.path.abspath(args.datapath)
    
    for idx in range(args.cycle_times):    
        if args.seed is None:
            seed = random.randint(0,10000)
        else:
            seed = args.seed
        Quant_MIA()
    

