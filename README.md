# Quant-MIA

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

Quant-MIA is a deep learning framework designed for efficient model quantization and Membership Inference Attacks (MIA).
It supports multiple quantization methods, including OBC, BRECQ, and AdaRound, and provides tools for evaluating privacy risks on different datasets under attack scenarios.

## Table of Contents   
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Results](#results)  
- [Acknowledgment](#acknowledgment)  
- [License](#license)  

## Usage  

### Quick Start 
```bash
cd Codes/
python main_workflow.py cifar10 path/to/dataset --auto_last_8bit --quant_range 2
```
This command will train a ResNet18 model and 64 shadow models on the selected dataset, then apply asymmetric channel-wise weight 2-bit quantization. Finally, it will perform MIA and save the attack results in `Codes/outdata.txt`.

### Running with Custom Arguments
You can customize the training and quantization process by modifying the `/Codes/config.yaml` file. After making changes, use the `--use_config` argument to load the updated settings. The project will apply all non-null values from the configuration file.

Warning: Be cautious when modifying keys that are set to null, as changing them incorrectly may cause conflicts with `main_workflow.py`. Unless you are very familiar with the code structure, it is not recommended to alter these values.

Example of running with a configuration file: 
```bash
python main_workflow.py cifar10 path/to/dataset --use_config
```

## Project Structure  
```markdown
Quant-MIA  
 ├── AdaRound/           # AdaRound quantization method  
 ├── BRECQ/              # BRECQ quantization method  
 ├── OBC/                # OBC quantization method  
 ├── Utils/              # Dataset preparation, dataloaders, evaluation, and result saving  
 ├── train.py            # Train the full-precision model  
 ├── plot.py             # Apply MIA and save results  
 ├── main_workflow.py    # Main script of the project  
 ├── README.md           # Project documentation  
 └── LICENSE             # License file  
```

## Results  
Below are example results obtained using the quick start command. The results can also be found in `Code/outdata.txt`.

```
Name Full Prec Attack Online
   AUC 0.7233, Accuracy 0.6382, TPR@0.1%FPR of 0.0696
Name Full Prec Attack Online fixed
   AUC 0.7236, Accuracy 0.6372, TPR@0.1%FPR of 0.0753
Name Full Prec Attack Offline
   AUC 0.5224, Accuracy 0.5501, TPR@0.1%FPR of 0.0160
Name Full Prec Attack Offline fixed
   AUC 0.5374, Accuracy 0.5533, TPR@0.1%FPR of 0.0420
Name Full Prec Attack Global threshold
   AUC 0.6230, Accuracy 0.6188, TPR@0.1%FPR of 0.0009
Name AdaRound w2_quant Attack Online
   AUC 0.7193, Accuracy 0.6362, TPR@0.1%FPR of 0.0770
Name AdaRound w2_quant Attack Online fixed
   AUC 0.7234, Accuracy 0.6362, TPR@0.1%FPR of 0.0783
Name AdaRound w2_quant Attack Offline
   AUC 0.5285, Accuracy 0.5546, TPR@0.1%FPR of 0.0183
Name AdaRound w2_quant Attack Offline fixed
   AUC 0.5428, Accuracy 0.5573, TPR@0.1%FPR of 0.0413
Name AdaRound w2_quant Attack Global threshold
   AUC 0.6234, Accuracy 0.6176, TPR@0.1%FPR of 0.0010
Name BRECQ w2_quant Attack Online
   AUC 0.7067, Accuracy 0.6281, TPR@0.1%FPR of 0.0560
Name BRECQ w2_quant Attack Online fixed
   AUC 0.7093, Accuracy 0.6278, TPR@0.1%FPR of 0.0716
Name BRECQ w2_quant Attack Offline
   AUC 0.5119, Accuracy 0.5360, TPR@0.1%FPR of 0.0121
Name BRECQ w2_quant Attack Offline fixed
   AUC 0.5262, Accuracy 0.5419, TPR@0.1%FPR of 0.0272
Name BRECQ w2_quant Attack Global threshold
   AUC 0.5993, Accuracy 0.6025, TPR@0.1%FPR of 0.0007
Name OBC w2_quant Attack Online
   AUC 0.7019, Accuracy 0.6241, TPR@0.1%FPR of 0.0516
Name OBC w2_quant Attack Online fixed
   AUC 0.7081, Accuracy 0.6265, TPR@0.1%FPR of 0.0671
Name OBC w2_quant Attack Offline
   AUC 0.5131, Accuracy 0.5323, TPR@0.1%FPR of 0.0104
Name OBC w2_quant Attack Offline fixed
   AUC 0.5246, Accuracy 0.5370, TPR@0.1%FPR of 0.0230
Name OBC w2_quant Attack Global threshold
   AUC 0.5988, Accuracy 0.6034, TPR@0.1%FPR of 0.0012
```

## Acknowledgment
This project includes code from:
- [OBC](https://github.com/IST-DASLab/OBC) 
- [BRECQ](https://github.com/yhhhli/BRECQ)
- [Lira-pytorch](https://github.com/orientino/lira-pytorch)

We have made modifications, including:

- Updating dataset handling functions
- Adding evaluation methods
- Implementing 1.58-bit model quantization

The original license terms apply to these portions of the code.

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

