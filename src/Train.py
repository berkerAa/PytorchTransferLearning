from __future__ import print_function, division
from __future__ import absolute_import
import pandas as pd
from packaging import version
from six.moves import range
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
from data_load import load_dataset
import data_split
import ReadParams
import socket
class Train():
        def __init__(self):
                self.Root = os.getcwd()
                self.Params = ReadParams.Params(os.path.join(self.Root, 'params', 'params.json'))
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #self.writer = SummaryWriter(self.Params.LogPath)
                self.DataSet = os.path.join(self.Root, self.Params.DataSetPath)
                self.Splitter = data_split.Splitter(0.2, self.DataSet)
                self.ModelLog = self.Params.ModelOutPath
                self.BoardLog = self.Params.LogPath
                self.is_inception = True
                self.InnerModel = None
                self.InnerLog = None
                self.SymDataset = None
                self.DataLoaders = None
                self.ClassNames = None
                self.DatasetSize = None
                self.ModelFt = None
                self.Writer = None
                self.OptimizerFt = None
                self.num_ftrs = None
                self.Criterion = None
                self.LrSchedule = None
                
        def StartLoop(self):
                for i in range(self.Params.TrainRep):
                        self.Iter(i)
                        self.ModelFt = None
                        self.CoolDown()
                        self.Writer.close()
        def Iter(self, _trainNum):
                self.Splitter.Run()
                self.SymDataset = self.Splitter.outDataset
                self.DataLoaders, self.ClassNames, self.DatasetSize = load_dataset(self.SymDataset, self.Params.ImgSize, self.Params.CropSize, self.Params.BatchSize)
                self.CreateStruct()
                try:
                    os.mkdir(self.InnerModel.format(_trainNum))
                except:
                    pass
                self.ModelLog = self.InnerModel.format(_trainNum)
                self.BoardLog = self.InnerLog.format(_trainNum)
                self.Preperations()
                self.runTensorboard()
                self.Train()
                self.killTensorboard()
        def Preperations(self):
                self.Writer = SummaryWriter(self.BoardLog)
                print('Downloading network...')
                self.ModelFt = eval('models.{}(pretrained={}, progress=True)'.format(self.Params.Model, True))
                if self.Params.Model == 'inception_v3':
                        self.ModelFt.aux_logits=True
                        self.ModelFt.AuxLogits.fc = nn.Linear(768, len(self.ClassNames))
                        self.is_inception = True
                print('Prepearing model architecture...')
                self.num_ftrs = self.ModelFt.fc.in_features
                self.ModelFt.fc = nn.Linear(self.num_ftrs, len(self.ClassNames))
                self.Criterion = nn.CrossEntropyLoss()
                self.OptimizerFt = self.getOptimizer()
                self.ModelFt = self.ModelFt.to(self.device)
                images, labels = next(iter(self.DataLoaders['train']))
                self.Writer.add_graph(self.ModelFt, images.to(self.device))
                self.LrSchedule = lr_scheduler.StepLR(self.OptimizerFt, step_size=self.Params.StepSize, gamma=self.Params.Gama)
                self.toFloatPoint16()
        def Train(self):
                since = time.time()
                best_model_wts = copy.deepcopy(self.ModelFt.state_dict())
                best_acc = 0.0
                for epoch in range(self.Params.EpochNum):
                        print('Epoch {}/{}'.format(epoch, self.Params.EpochNum - 1))
                        print('-' * 10)
                        print('Current Gpu Tempreture is:', self.GetGpuTemp())
                        if self.GetGpuTemp() >= 75:
                                while self.GetGpuTemp() >= 60:
                                        print('Halting training proccess for cool down')
                                        time.sleep(2*60)
                                        print('Current Gpu Tempreture is:', self.GetGpuTemp())
                        confusion_matrix = torch.zeros(len(self.ClassNames), len(self.ClassNames))
                        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
                        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')        
                        # Each epoch has a training and validation phase
                        for phase in ['train', 'val']:
                                if phase == 'train':
                                        self.ModelFt.train()  # Set model to training mode
                                else:
                                        self.ModelFt.eval()   # Set model to evaluate mode
                                        #self.resnext50_32x4d(epoch)
                                running_loss = 0.0
                                running_corrects = 0
                                for inputs, labels in self.DataLoaders[phase]:
                                        inputs = inputs.to(self.device).half()
                                        labels = labels.to(self.device)

                                        # zero the parameter gradients
                                        self.OptimizerFt.zero_grad()

                                        # forward
                                        # track history if only in train
                                        with torch.set_grad_enabled(phase == 'train'):
                                            if self.is_inception and phase == 'train':
                                                outputs, aux_outputs = self.ModelFt(inputs)
                                                loss1 = self.Criterion(outputs, labels)
                                                loss2 = self.Criterion(aux_outputs, labels)
                                                loss = loss1 + 0.4*loss2
                                            else:
                                                outputs = self.ModelFt(inputs)                    
                                                loss = self.Criterion(outputs, labels)
                                            _, preds = torch.max(outputs, 1)
                                            # backward + optimize only if in training phase
                                            if phase == 'train':
                                                loss.backward()
                                                self.OptimizerFt.step()
                                            else:
                                               predlist=torch.cat([predlist,preds.view(-1).cpu()])
                                               lbllist=torch.cat([lbllist,labels.view(-1).cpu()]) 
                                        # statistics
                                        running_loss += loss.item() * inputs.size(0)
                                        running_corrects += torch.sum(preds == labels.data)

                                if phase == 'train':
                                        self.LrSchedule.step()

                                epoch_loss = running_loss / self.DatasetSize[phase]
                                epoch_acc = running_corrects.double() / self.DatasetSize[phase]

                                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                                        phase, epoch_loss, epoch_acc))
                                if phase == 'train':
                                        self.Writer.add_scalar('Loss/train', epoch_loss, epoch)
                                        self.Writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                                else:
                                        self.Writer.add_scalar('Gpu/temp', self.GetGpuTemp(), epoch)
                                        self.Writer.add_scalar('Loss/val', epoch_loss, epoch)
                                        self.Writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                                if phase == 'val' and epoch_acc > best_acc:
                                        best_acc = epoch_acc
                                        best_model_wts = copy.deepcopy(self.ModelFt.state_dict())
                                        conf_mat=sklearn.metrics.confusion_matrix(lbllist.numpy(), predlist.numpy())
                                        con_mat = np.asarray(conf_mat)
                                        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
                                        df_cm = pd.DataFrame(con_mat_norm, range(100), range(100))
                                        plt.figure(figsize=(16,16))
                                        sn.set(font_scale=1.4) # for label size
                                        sns_plot = sn.heatmap(df_cm, annot=False)
                                        self.Writer.add_figure("Confusion Matrix", sns_plot.figure, global_step=epoch)
                                        
                                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.ModelFt.state_dict(),
                                    'optimizer_state_dict': self.OptimizerFt.state_dict(),
                                    'loss': epoch_loss
                                    }, '{}/Resnext{}_acc{}.pth'.format(self.ModelLog, epoch, best_acc))
                                print()
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
                print('Best val Acc: {:4f}'.format(best_acc))
        def toFloatPoint16(self):
                self.ModelFt.half()  # convert to half precision
                for layer in self.ModelFt.modules():
                        if isinstance(layer, nn.BatchNorm2d):
                                layer.float()
        def getOptimizer(self):
                if self.Params.Optimizer=='SGD':
                        return eval('optim.{}(self.ModelFt.parameters(), lr={}, momentum={})'.format(self.Params.Optimizer, self.Params.LearningRate, self.Params.Momentum))
                elif self.Params.Optimizer=='Adam':
                        return eval('optim.{}(self.ModelFt.parameters(), lr={})'.format(self.Params.Optimizer, self.Params.LearningRate))

        def GetGpuTemp(self):
                return int(os.popen("nvidia-smi -q | grep 'GPU Current Temp' | cut -d' ' -f 24").read().replace('\n', ''))

        def CoolDown(self):
                print('Starting sleep loop for cool down hardware...')
                while self.GetGpuTemp() > 50:
                        print('Gpu Tempruture is : ', self.GetGpuTemp())
                        time.sleep(5 * 60)  
                        
        def CreateStruct(self):
                if self.Params.CreateStructure :
                        self.Splitter.CreateStructure()
                try:
                        os.mkdir(self.ModelLog)
                        os.mkdir(self.BoardLog)
                except:
                        pass
                self.InnerModel = os.path.join(self.ModelLog,'{}')
                self.InnerLog = os.path.join(self.BoardLog, '{}')
        def getIp(self):
                hostname = socket.gethostname()    
                return socket.gethostbyname(hostname)
        def runTensorboard(self):
                os.popen('tensorboard  --logdir {} --host {} --port 6006'.format(self.BoardLog, self.getIp(), 6006))
        def killTensorboard(self):
                os.popen('pkill tensorboard')
        def resnext50_32x4d(self, epoch):
                c = 0
                for layer in self.ModelFt.layer1:
                        self.Writer.add_histogram("{}/conv1".format('layer1'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer1'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer1'), layer.conv2.weight, epoch)
                for layer in self.ModelFt.layer2:
                        self.Writer.add_histogram("{}/conv1".format('layer2'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer2'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer2'), layer.conv2.weight, epoch)
                for layer in self.ModelFt.layer3:
                        self.Writer.add_histogram("{}/conv1".format('layer3'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer3'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer3'), layer.conv2.weight, epoch)
                for layer in self.ModelFt.layer4:
                        self.Writer.add_histogram("{}/conv1".format('layer4'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer4'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer4'), layer.conv2.weight, epoch)
                
                self.Writer.add_histogram("FC/weight",self.ModelFt.fc.weight, epoch)
                self.Writer.add_histogram("FC/bias",self.ModelFt.fc.bias, epoch)
if __name__ == '__main__':
        obj = Train()
        obj.StartLoop()
