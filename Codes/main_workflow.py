import argparse
import subprocess
import random
import os

work_dir = os.path.abspath(os.path.dirname(__file__))
quant_methods = {'AdaRound':'main_quant.py', 'BRECQ':'main_imagenet.py', 'OBC':'main_trueobs.py'}
bit_str = {1:'1', 1.58:'log3', 2:'2', 3:'3', 4:'4'}
datapath = ''
seed = 0
skip_shadows = False

def shadow_models_trainer():
    global skip_shadows
    if skip_shadows:
        return
    for shadow_id in range(args.n_shadows):
        command = [
                'python', 
                os.path.join(work_dir, 'train.py'), 
                '--dataset', args.dataset, 
                '--datapath', datapath, 
                '--seed', f'{random.randint(0,10000)}',
                '--n_shadows', f'{args.n_shadows}',
                '--shadow_id', f'{shadow_id}',
                '--savedir', os.path.join(work_dir, 'shadows', f'{args.dataset}')
            ]
        if args.use_config:
            command.append('--config')
        subprocess.run(command,check=True)
    skip_shadows = True

def target_model_trainer(pkeep = 0.5):
    command = [
            'python', 
            os.path.join(work_dir, 'train.py'), 
            '--dataset', args.dataset, 
            '--datapath', datapath, 
            '--seed', f'{seed}',
            '--savedir', os.path.join(work_dir, 'models', f'{args.dataset}'),
            '--pkeep', f'{pkeep}'
        ]
    if args.use_config:
        command.append('--config')
    subprocess.run(command,check=True)

    
def quantizer():
    for method in quant_methods.keys():    
        if method == 'AdaRound':
            continue
        for bit in args.quant_range: 
            command = [
                    'python',
                    os.path.join(work_dir, method, quant_methods[method]),
                    '--dataset', args.dataset,
                    '--datapath', datapath,
                    '--seed', f'{seed}',
                    '--load', os.path.join(work_dir, 'models', f'{args.dataset}', 'model.pt'),
                    '--keep', os.path.join(work_dir, 'models', f'{args.dataset}', 'keep.npy'),
                    '--wbits', f'{bit}',
                    '--save', os.path.join(work_dir, 'quant_models', f'{args.dataset}', method, f'w{bit_str[bit]}_quant','model.pt'),
                    '--asym', f'{not args.sym}'
                ]
            
            if args.use_config:
                command.append('--config')
            
            if args.auto_last_8bit and (args.dataset == 'tiny-imagenet' or bit < 2):
                command.append('--last_layer_8bit')
            elif args.last_layer_8bit:
                command.append('--last_layer_8bit')
                
            if args.use_config:
                command.append('--config')
            
            subprocess.run(command,check=True)
            
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
                        '--name', f'{legend} {dirname}'
                    ],check=True
                )
                
def Quant_MIA():
    # target_model_trainer()
    shadow_models_trainer()
    quantizer()
    plot()    
    
def test_pkeep(pkeep_list):
    for pkeep in pkeep_list:
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
    parser.add_argument(
        '--auto_last_8bit', help='Use 8bit quantization for last layer', action='store_true')
    parser.add_argument(
        '--pkeeps', help='Test pkeep', type=float, nargs='+' ,default=None)
    parser.add_argument(
        '--skip_shadows', help='Skip training shadow models', action='store_true')
    parser.add_argument(
        '--last_layer_8bit', help='Use 8bit quantization for last layer, make sure auto_last_8bit is false', action='store_true')

    args = parser.parse_args()    
    datapath = os.path.abspath(args.datapath)
    
    if args.pkeeps is not None:
        test_pkeep(args.pkeeps)
        exit()
        
    if args.skip_shadows:
        skip_shadows = True
    
    for idx in range(args.cycle_times):    
        if args.seed is None:
            seed = random.randint(0,10000)
        else:
            seed = args.seed
        Quant_MIA()
    

