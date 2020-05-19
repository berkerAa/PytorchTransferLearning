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
#### Installing Necessary Libraries
This project mostly depended to pytorch so to install, please head to **[pytorch official website](https://pytorch.org/)** for install latest stable release. <br/>
If you want you monitor your real time progress using Tensorboard along with Confusion Matrix, There is few packages nedded to be install. **sklearn** library for creating confusion matrix, **pandas** library for creating temporary data frame object in order to creating heatmap and **seaborn** library for creating confusion matrix heatmap. For installation, please run:
```
pip3 install scikit-learn, seaborn, pandas
```
#### Dataset Structure
Dataset split script was hardcoded for specific structure type as follows: <br/>
```
      **DATASET/** <br/>
        .....**ClassName/** <br/>
                      ....**Image1** <br/>
                      ....**Image2** <br/>
                      ....**ImageN** <br/>
        .....**ClassName/** <br/>
                      ....**Image1** <br/>
                      ....**Image2** <br/>
                      ....**ImageN** <br/>
        .....**ClassName/** <br/>
                      ....**Image1** <br/>
                      ....**Image2** <br/>
                      ....**ImageN** <br/>
```
Please make sure your structure fits above constraints.                      
#### Dataset Split Script Details

#### Changeable paramters using params.json file
#### Train Script Details and Changing Specific Functions Based on Running System
# Example Usage on UECFOOD Dataset
# Authors
# License
# Acknowledgments

