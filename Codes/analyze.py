import re
import numpy as np
from collections import defaultdict
import argparse
import os

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
        
        data[quant][wbits][attack] = {
            "AUC": [],
            "Accuracy": [],
            "TPR": []
        }
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
                
print(f"Results:{results}")
 

