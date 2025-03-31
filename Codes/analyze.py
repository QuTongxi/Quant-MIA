import re
import numpy as np
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Analyze AUC, Accuracy, and TPR from log files.")
parser.add_argument(
    "--metrics",
    type=str,
    default="outdata.txt",
    help="Path to the log file containing AUC, Accuracy, and TPR data.",
)
parser.add_argument(
    "--accuracy",
    type=str,
    default="output.txt",
    help="Path to the log file containing train and test accuracy.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    help="the dataset where you test the accuracy on",
)
args = parser.parse_args()

data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'AUC': [], 'Accuracy': [], 'TPR': []})))

with open(args.metrics, "r", encoding="utf-8") as file:
    lines = file.readlines()

current_key = None

for line in lines:
    line = line.strip()
    if line.startswith("Name"):  # 解析 Name 和 Attack 组成的 key
        pattern = r"Name\s+(.*?)\s+Attack\s+(.*)"
        match = re.search(pattern, line)
        
        quant = match.group(1).split(' ')[0]
        wbits = match.group(1).split(' ')[1]
        wbits = wbits.split('_')[0]
        attack = match.group(2)

        current_key = f"{quant} {wbits} {attack}"
    else:
        # 匹配 AUC, Accuracy, TPR 值
        match = re.search(r"AUC ([\d.]+), Accuracy ([\d.]+), TPR@0.1%FPR of ([\d.]+)", line)
        if match and current_key:
            auc, accu, tpr = map(float, match.groups())
            quant, wbits, attack = current_key.split(' ', maxsplit=2)
            
            data[quant][wbits][attack]["AUC"].append(auc)
            data[quant][wbits][attack]["Accuracy"].append(accu)
            data[quant][wbits][attack]["TPR"].append(tpr)

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'AUC_mean': 0, 'Accuracy_mean': 0, 'TPR_mean': 0, 'AUC_std': 0, 'Accuracy_std': 0, 'TPR_std': 0})))
for quant, wbits_dict in data.items():
    for wbits, attack_dict in wbits_dict.items():
        for attack, metrics in attack_dict.items():
            for metric, values in metrics.items():
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    results[quant][wbits][attack][f"{metric}_mean"] = mean_value
                    results[quant][wbits][attack][f"{metric}_std"] = std_value
                else:
                    raise ValueError(f"No values found for {quant}, {wbits}, {attack}, {metric}")
                
for quant, wbits_dict in results.items():
    for wbits, attack_dict in wbits_dict.items():
        for attack, metrics in attack_dict.items():
            print(f"Quant: {quant}, Wbits: {wbits}, Attack: {attack}")
            print(f"AUC Mean: {metrics['AUC_mean']:.4f}, AUC Std: {metrics['AUC_std']:.4f}")
            print(f"Accuracy Mean: {metrics['Accuracy_mean']:.4f}, Accuracy Std: {metrics['Accuracy_std']:.4f}")
            print(f"TPR Mean: {metrics['TPR_mean']:.4f}, TPR Std: {metrics['TPR_std']:.4f}")
            print("-" * 50)
                


def plot_grouped_bar_chart(data, full_prec, save_path):
    categories = list(data.keys())  
    categories = sorted(categories, reverse=True)  # Sort categories by wbits
    subcategories = list(next(iter(data.values())).keys())  
    subcategories = sorted(subcategories)  # Sort subcategories by wbits
    metrics = [key.replace('_mean', '') for key in next(iter(next(iter(data.values())).values())).keys() if '_mean' in key]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), sharey=False)

    if len(metrics) == 1:
        axes = [axes]

    bar_width = 0.15
    font_size = 16
    x_positions = np.arange(len(categories))  

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for k, subcategory in enumerate(subcategories):
            means = [data[category][subcategory][f'{metric}_mean'] for category in categories]  
            stds = [data[category][subcategory][f'{metric}_std'] for category in categories]  

            ax.bar(x_positions + k * bar_width, means, width=bar_width, yerr=stds, capsize=5, 
                   color=colors[k], edgecolor='black', label=f'{subcategory}')

        ax.set_xticks(x_positions + bar_width * (len(subcategories) - 1) / 2)
        ax.set_xticklabels(categories, fontsize=font_size)
        ax.set_xlabel('wbits', fontsize=font_size)
        ax.set_ylabel(metric, fontsize=font_size)
        ax.set_title(f'{metric} for quantized models', fontsize=font_size)
        ax.legend(title='Methods',loc='lower left')
        ax.set_ylim(bottom=0)

        # full_prec 参考线
        mean_value = full_prec[f'{metric}_mean']
        std_value = full_prec[f'{metric}_std']

        ax.axhline(y=mean_value, color='gray', linestyle='dashed', linewidth=1)  # 水平虚线
        ax.fill_between([-0.5, len(categories) - 0.5], mean_value - std_value, mean_value + std_value, 
                        color='gray', alpha=0.2)  # 浅色覆盖区域

        # ax.text(len(categories) - 0.5, mean_value, f'{mean_value:.2f}', 
        #         verticalalignment='bottom', horizontalalignment='right', fontsize=font_size, color='gray')

    plt.tight_layout()
    plt.savefig(save_path)


def plot_results(result_dict, mode, fixed=False):
    """
    Generalized function to plot results for online/offline modes with or without fixed settings.

    Args:
        result_dict (dict): The dictionary containing the results.
        mode (str): Either 'Online' or 'Offline'.
        fixed (bool): Whether to use the 'fixed' version of the mode.
    """
    from collections import defaultdict

    # Initialize dictionaries
    plot_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    full_prec = {}

    # Determine the key suffix based on the 'fixed' flag
    mode_key = f"{mode} fixed" if fixed else mode

    # Populate the plot dictionary
    for quant, wbits_dict in result_dict.items():
        for wbits, attack_dict in wbits_dict.items():
            if wbits == 'wlog3':
                plot_dict['w1.58'][quant] = attack_dict[mode_key]
            elif wbits == 'Prec':
                full_prec = attack_dict[mode_key]
            else:
                if wbits == 'w3':
                    continue
                plot_dict[wbits][quant] = attack_dict[mode_key]

    # Generate the plot
    save_path = f"plots/{mode.lower()}{'_fixed' if fixed else ''}_plot.png"
    plot_grouped_bar_chart(plot_dict, full_prec, save_path)

def parse_log_file(file_path):
    full_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    quant_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    current_method = None
    current_dataset = None
    current_pkeep = None
    current_wbits = None
    
    for line in lines:
        # Match Full model line
        full_match = re.match(r'\[Full\] dataset: ([\w-]+) epochs: \d+ pkeep: ([\d.]+)', line)
        if full_match:
            current_method = 'Full'
            current_dataset = full_match.group(1)
            current_pkeep = full_match.group(2)
            continue
        
        # Match Quantized model line
        quant_match = re.match(r'\[(\w+)\] dataset: ([\w-]+) wbits: ([\d.]+)', line)
        if quant_match:
            current_method = quant_match.group(1)
            current_dataset = quant_match.group(2)
            current_wbits = quant_match.group(3)
            continue
        
        # Match accuracy lines
        accu_match = re.match(r'Train Accu.: ([\d.]+) Test Accu.: ([\d.]+)', line)
        if accu_match:
            train_acc = float(accu_match.group(1))
            test_acc = float(accu_match.group(2))
            
            if current_method == 'Full':
                full_dict[current_dataset][current_pkeep]['train'].append(train_acc)
                full_dict[current_dataset][current_pkeep]['test'].append(test_acc)
            else:
                quant_dict[current_dataset][current_method][current_wbits]['train'].append(train_acc)
                quant_dict[current_dataset][current_method][current_wbits]['test'].append(test_acc)
    
    return dict(full_dict), dict(quant_dict)


def get_mean_for_accu(data:dict):
    for k1, v1 in data.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                if isinstance(v3, dict):
                    # quant_dict
                    for k4, v4 in v3.items():
                        data[k1][k2][k3][k4] = {'mean' : np.mean(v4), 'std' : np.std(v4)}
                else:
                    data[k1][k2][k3] = {'mean' : np.mean(v3), 'std' : np.std(v3)}
                    
    return data

def prepare_for_accuracy_plot(full, quant, dataset, pkeep='0.5'):
    plot_args = {'methods':[], 'bits':[], 'mean':{}, 'std':{}, 'full_mean':0.0, 'full_std':0.0}
    plot_args['methods'] = list(quant[dataset].keys())
    plot_args['bits'] = list(next(iter(quant[dataset].values())).keys())
    
    for method, v1 in quant[dataset].items():
        mean_data, std_data = [], []
        for bit, v2 in v1.items():
            mean_data.append(v2['test']['mean'])
            std_data.append(v2['test']['std'])
            
        plot_args['mean'][method] = mean_data
        plot_args['std'][method] = std_data
        
    plot_args['full_mean'] = full[dataset][pkeep]['test']['mean']
    plot_args['full_std'] = full[dataset][pkeep]['test']['std']
    return plot_args
        
def plot_accuracy(args, save_path='plots/accuracy_plot.png'):
    methods = args['methods']
    bits = args['bits']
    test_means = args['mean']
    test_stds = args['std']

    full_mean = args['full_mean']
    full_std = args['full_std']
    
    y_min = min(min(min(values) for values in test_means.values()), full_mean) - 5
    y_max = max(max(max(values) for values in test_means.values()), full_mean) + 5
    
    y_min = max(y_min, 0)
    y_max = min(y_max, 100)
    
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.errorbar(bits, test_means[method], yerr=test_stds[method], label=method,
                    marker='o', capsize=5, linestyle='-', alpha=0.8)

    plt.axhline(y=full_mean, color='r', linestyle='--', label='FULL Mean')
    plt.fill_between(bits, full_mean - full_std, full_mean + full_std, color='r', alpha=0.2, label='FULL Std')

    plt.xlabel('Bits')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Quantization Test Accuracy with Error Bars')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(y_min, y_max)
    
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(save_path, dpi=300)


def print_test_accuracy(data:dict):
    for k1, v1 in data.items():
        print(f'{k1}')
        for k2, v2 in v1.items():
            print(f'  {k2}')
            for k3, v3 in v2.items():
                print(f'    {k3}', end='')                
                if isinstance(next(iter(v3.values())), dict):
                    # quant_dict
                    print()
                    for k4, v4 in v3.items():
                        print(f'    {k4} : {data[k1][k2][k3][k4]['mean']}  |  {data[k1][k2][k3][k4]['std']}')
                else:
                    print(f': {data[k1][k2][k3]['mean']}  |  {data[k1][k2][k3]['std']}')
                    
                        
if __name__ == "__main__":
    
    os.makedirs('plots', exist_ok=True)
    
    plot_results(results, 'Online')
    plot_results(results, 'Offline')
    plot_results(results, 'Online', fixed=True)
    plot_results(results, 'Offline', fixed=True)
    
    full, quant = parse_log_file(args.accuracy)
    
    print(full['cifar100']['0.5']['test'])
    
    full = get_mean_for_accu(full)
    quant = get_mean_for_accu(quant)
    
    # full[<dataset>][<pkeep>][train/test]
    # quant[<dataset>][<method>][<wbits>][train/test]
    def print_accu():
        print('-'*25, 'full', '-'*25)
        print_test_accuracy(full)
        print('-'*25, 'quant', '-'*25)
        print_test_accuracy(quant)
    # print_accu()  
    
    plot_args = prepare_for_accuracy_plot(full, quant, dataset=args.dataset)
    plot_accuracy(plot_args)  