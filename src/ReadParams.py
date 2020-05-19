import json
class Params:
        def __init__(self, path):
                self.JsonFile = None
                self.Read(path)
                self.LogPath = self.GetVal('Log path')
                self.CreateStructure = int(self.GetVal('Create Structure'))
                self.Model = self.GetVal('Pretrained Model')
                self.ModelOutPath = self.GetVal('Model output path')
                self.TrainRep = int(self.GetVal('Train repitation'))
                self.EpochNum = int(self.GetVal('Epochs'))
                self.StepSize = int(self.GetVal('Step size'))
                self.Gama = float(self.GetVal('Gama'))
                self.Momentum = float(self.GetVal('Momentum'))
                self.ImgSize = int(self.GetVal('Image Size'))
                self.CropSize = int(self.GetVal('Cropped Image Size'))
                self.DataSetPath = self.GetVal('Dataset path')
                self.BatchSize = int(self.GetVal('Batch Size'))
                self.LearningRate = float(self.GetVal('Learning Rate'))
                self.Optimizer = self.GetVal('Optimizer')
                self.Monitor = int(self.GetVal('Tensorboard'))                 
        def Read(self, path):
                with open(path) as json_file:
                        self.JsonFile = json.load(json_file)
        def GetVal(self, key):
                return self.JsonFile[key]
