import re
import numpy as np
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Analyze AUC, Accuracy, and TPR from log files.")
parser.add_argument(
    "--file_path",
    type=str,
    default="outdata.txt",
    help="Path to the log file containing AUC, Accuracy, and TPR data.",
)
args = parser.parse_args()

data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'AUC': [], 'Accuracy': [], 'TPR': []})))

with open(args.file_path, "r", encoding="utf-8") as file:
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
    
if __name__ == "__main__":
    
    os.makedirs('plots', exist_ok=True)
    
    plot_results(results, 'Online')
    plot_results(results, 'Offline')
    plot_results(results, 'Online', fixed=True)
    plot_results(results, 'Offline', fixed=True)