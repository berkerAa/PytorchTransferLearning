from torch.utils.tensorboard import SummaryWriter
import os
import socket
class Monitor:
    def __init__(self, BoardLog):
        self.BoardLog = BoardLog
        self.Writer = SummaryWriter(self.BoardLog)
    def addModelGraph(self, torchModel, torchInput):
        self.Writer.add_graph(torchModel, torchInput)
    def addScalars(self, y, epoch, phase):
        self.Writer.add_scalar('Loss/{}'.format(phase), y, epoch)
        self.Writer.add_scalar('Accuracy/{}'.format(phase), y, epoch)
        if phase == 'val':
            self.Writer.add_scalar('Gpu/Temp', self.GetGpuTemp(), epoch)
    def addConf(self, figure, epoch):
        self.Writer.add_figure("Confusion Matrix", figure, global_step=epoch)
    def GetGpuTemp(self):
        return int(os.popen("nvidia-smi -q | grep 'GPU Current Temp' | cut -d' ' -f 24").read().replace('\n', ''))
    def runTensorboard(self):
                os.popen('tensorboard  --logdir {} --host {} --port {}'.format(self.BoardLog, self.getIp(), 6006))
    def killTensorboard(self):
                os.popen('pkill tensorboard')
    def resnext50_32x4d(self, ModelFt, epoch):
                for layer in ModelFt.layer1:
                        self.Writer.add_histogram("{}/conv1".format('layer1'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer1'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer1'), layer.conv2.weight, epoch)
                for layer in ModelFt.layer2:
                        self.Writer.add_histogram("{}/conv1".format('layer2'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer2'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer2'), layer.conv2.weight, epoch)
                for layer in ModelFt.layer3:
                        self.Writer.add_histogram("{}/conv1".format('layer3'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer3'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer3'), layer.conv2.weight, epoch)
                for layer in ModelFt.layer4:
                        self.Writer.add_histogram("{}/conv1".format('layer4'), layer.conv1.weight, epoch)
                        self.Writer.add_histogram("{}/Conv3".format('layer4'), layer.conv3.weight, epoch)
                        self.Writer.add_histogram("{}/Conv2".format('layer4'), layer.conv2.weight, epoch)
                
                self.Writer.add_histogram("FC/weight",ModelFt.fc.weight, epoch)
                self.Writer.add_histogram("FC/bias",ModelFt.fc.bias, epoch)
    def getIp(self):
                hostname = socket.gethostname()    
                return socket.gethostbyname(hostname)