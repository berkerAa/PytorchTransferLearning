from torchvision import models
import torch.nn as nn
class CreateModel:
    def __init__(self, Params, SoftMax):
        print('Downloading network...')
        self.SoftMax = SoftMax
        self.PretrainedModel = eval('models.{}(pretrained={}, progress=True)'.format(Params.Model, True))
        eval('self.{}()'.format(Params.Model))
    def resnet18(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def alexnet(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.classifier[6] = nn.Linear(self.num_ftrs, self.SoftMax)
    def vgg16(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.classifier[6] = nn.Linear(self.num_ftrs, self.SoftMax)
    def squeezenet1_1(self):
        self.PretrainedModel.classifier[1] = nn.Conv2d(512, self.SoftMax, kernel_size=(1,1), stride=(1,1))
    def densenet121(self):
        self.num_ftrs = self.PretrainedModel.classifier.in_features
        self.PretrainedModel.classifier = nn.Linear(self.num_ftrs, self.SoftMax)
    def inception_v3(self):
        self.PretrainedModel.aux_logits=True
        self.PretrainedModel.AuxLogits.fc = nn.Linear(768, self.SoftMax)
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def googlenet(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def shufflenet_v2_x1_0(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.classifier[1] = nn.Conv2d(self.num_ftrs, self.SoftMax, kernel_size=(1,1), stride=(1,1))
    def mobilenet_v2(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def resnext50_32x4d(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def wide_resnet50_2(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
    def mnasnet1_3(self):
        self.num_ftrs = self.PretrainedModel.fc.in_features
        self.PretrainedModel.fc = nn.Linear(self.num_ftrs, self.SoftMax)
