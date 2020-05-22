# PytorchTransferLearning
a Transfer Learning project for simplifying training process using **pytorch** framework and **torchvision.models** database originally created for UECFood100 classification task
# Table of Contents
- [Getting Started](#getting-started)
  - [Installing Necessary Libraries](#installing-necessary-libraries)
  - [Dataset Structure](#dataset-structure)
  - [Dataset Split Script Details](#dataset-split-script-details)
  - [Changeable paramters using params.json file](#changeable-parameters-using-params.json-file)
  - [Train Script Details and Changing Specific Functions Based on Running System](#train-script-details-and-changing-specific-functions-based-on-running-system)
- [Example Usage on Fake Face Classification Task](#example-usage-on-fake-face-classification-task)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)
# Getting Started
These instructions will get you a copy of the project up and running on your local machine for train and testing purposes. See [Example Usage](#example-usage-on-uecfood-dataset) for notes on how to deploy the project and run a demo on UECFOOD Dataset. Before get started, please be sure that Cuda and Nvidia drivers correctly installed and added to bash for avoid further runtime errors.
## Installing Necessary Libraries
This project mostly depended on pytorch so for installation, please head to **[pytorch official website](https://pytorch.org/)** for install latest stable release. <br/>
If you want you monitor your real time progress using Tensorboard along with Confusion Matrix, There is few packages nedded to be installed. **sklearn** library for creating confusion matrix, **pandas** library for creating temporary data frame object in order to creating heatmap and **seaborn** library for creating confusion matrix heatmap. For installation, please run:
```
pip3 install scikit-learn, seaborn, pandas
```
## Dataset Structure
Dataset split script was hardcoded for specific structure type as follows: <br/>
```
      DATASET/ 
              .....ClassName/ 
                            ....Image1 
                            ....Image2 
                            ....ImageN 
              .....ClassName/ 
                            ....Image1 
                            ....Image2 
                            ....ImageN 
              .....ClassName/ 
                            ....Image1 
                            ....Image2 
                            ....ImageN 
```
Please make sure your structure fits above constraints.                      
## Dataset Split Script Details
Dataset Split scirpt aims in creating the training and validation folder structure for pytorch dataloader object with random 20% validation divison per classes. For avoiding unnecessary read and write operations, this script creates symlinks of generated train and validation image paths to a symlink folder named **SYMDATASET** and generates a json fomratted file for saving related image paths into two headers named **Train** and **Test** for further usage. <br/>
***BE CAUTIOUS WHEN USING THIS SCRIPT ON WINDOWS OPERATING SYSTEMS.*** This script was originally created and tested on Ubuntu 18.04 operating system and used os.remove() builtin function for unlink pre created symlink on related SYMDATASET folder. It seems like this function may delete original paths where symlinks directs. For more information, please head to [stackoverflow](https://stackoverflow.com/questions/11700545/how-to-delete-a-symbolic-link-in-python) 
## Changeable parameters using params.json file
**Log Path**: Desired path for save Tensorboard log files, default: runs/resnext50_32x4d <br/>
**Create Structure**: Checks if [CreateStructure](https://github.com/berkerAa/PytorchTransferLearning/blob/4181536e397656d79d14e8e989f5b451a676aa20/src/data_split.py#L18-L30) function will run.Change to 1 for first use, default: 0 <br/>
**Pretrained Model**: Model name from **[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)** for pretrained model weights, default: resnext50_32x4d <br/>
**Model output path**: Desired path for save Model outputs, default: Models/resnext50_32x4d <br/>
**Train repitation**: Int value for how many trains be done with same parameters and different splits before kill process, default: 5 <br/>
**Dataset path**: Dataset path that fits specifications. For more details, please head to [Dataset Structure](#dataset-structure). Default: cropped_names <br/>
For more parameter settings and example params.json files, please head to [params](https://github.com/berkerAa/PytorchTransferLearning/tree/master/params). Before run Train.py script, be sure that desired parameters file is named like params.json. For example if you want to train your network according to [this](https://github.com/berkerAa/PytorchTransferLearning/blob/master/params/densenet.params.json) parameters, rename file as params.json then run the Train script.
## Train Script Details and Changing Specific Functions Based on Running System
Train.py script uses src/Monitor.py, src/CreateModel.py, src/ReadParams.py scripts backhand according to given parameters. Please be sure that these files present for avoid problems. There is few things needed to be editted before start according to your local system. First of all, this script originally starts training process with mixed precision concept. If you want to disable mix precision feature please comment this line: [line87](https://github.com/berkerAa/PytorchTransferLearning/blob/79102c77dfe3a26087b41011c8f0f9ca7830de90/src/Train.py#L87) and delete .half() functions call on this line: [line114](https://github.com/berkerAa/PytorchTransferLearning/blob/79102c77dfe3a26087b41011c8f0f9ca7830de90/src/Train.py#L114). For more information about Mixed precision concept you can visit this [link](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). <br/>
We were added a cooldown condition that checks and acts according GPU Tempreture for avoid high hardware tempreture during training process. This may cause a slightly slow training process dependet on used systems. For disable this condition check, you can comment [lines](https://github.com/berkerAa/PytorchTransferLearning/blob/79102c77dfe3a26087b41011c8f0f9ca7830de90/src/Train.py#L96-L100).  
# Example Usage on Fake Face Classification Task
Before starting, please make sure you [install](#installing-necessary-libraries) mandatory libraries, changed realted parts on [Train](#train-script-details-and-changing-specific-functions-based-on-running-system) script. **This example designed and tested on Ubuntu 18.04 LTS operating system with Gpu Support, folowing unix commands may vary according to operating system. For Windows users, please head to [Dataset split section](#dataset-split-script-details) for more information.** We will be using Fake-Real face images dataset for this example, dataset can be downloaded via this [link](https://drive.google.com/open?id=1smby8vBB0g8bNtUsQ10OdjF-4FK9k_GS)  
# Authors
# License
# Acknowledgments

