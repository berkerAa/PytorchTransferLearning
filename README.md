# PytorchTransferLearning
a Transfer Learning project for simplifying training process using **pytorch** framework and **torchvision.models** database originally created for UECFood100 classification task
# Table of Contents
- [Getting Started](#getting-started)
  - [Installing Necessary Libraries](#installing-necessary-libraries)
  - [Dataset Structure](#dataset-structure)
  - [Dataset Split Script Details](#dataset-split-script-details)
  - [Changeable paramters using params.json file](#changeable-paramters-using-params.json-file)
  - [Train Script Details and Changing Specific Functions Based on Running System](#train-script-details)
- [Example Usage on UECFOOD Dataset](#example-usage-on-uecfood-dataset)
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
## Changeable paramters using params.json file
**Log Path**: Desired path for save Tensorboard log files, default: runs/Inception_Fp16_320
**Create Structure**: Checks if [CreateStructure](https://github.com/berkerAa/PytorchTransferLearning/blob/4181536e397656d79d14e8e989f5b451a676aa20/src/data_split.py#L18-L30) function will run.Change to 1 for first use, default: 0

## Train Script Details and Changing Specific Functions Based on Running System
# Example Usage on UECFOOD Dataset
# Authors
# License
# Acknowledgments

