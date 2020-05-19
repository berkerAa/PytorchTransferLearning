import numpy as np
import os
import sklearn.model_selection as model_selection
import json
class Splitter:
        def __init__(self, DivisonParam, DataSet):
        #DivisionParam::float object that holds test split value
        #DataSet::str object that holds original dataset path
        #PlaceHolder::Dict object that holds uniqe train,test symlink paths
        #outDataset, Train, Test::str object that holds Symlink parent folder paths
                self.DivisionParam = DivisonParam
                self.DataSet = DataSet
                self.PlaceHolder = {}
                self.Root = os.getcwd()
                self.outDataset = os.path.join(self.Root, 'SymDataset')
                self.Train = os.path.join(self.Root,self.outDataset, 'train')
                self.Test = os.path.join(self.Root,self.outDataset, 'val')
        def CreateStructure(self):
        #Creating output Sym folder structure with copying original Dataset folder structure:
        #Root
        #    ----Dataset
        #       -------Class1
        #       -------Class2
        #       -------ClassN
                os.mkdir(self.outDataset)
                os.mkdir(self.Train)
                os.mkdir(self.Test)
                for i in os.listdir(self.DataSet):
                        os.mkdir(os.path.join(self.Train, i))
                        os.mkdir(os.path.join(self.Test, i))
        def CheckOut(self):
        #Checks output folders if there is any pre defined symlink images inside
                TrainCount = len(os.listdir(os.path.join(self.Train, os.listdir(self.Train)[0])))
                TestCount = len(os.listdir(os.path.join(self.Test, os.listdir(self.Test)[0])))
                if TrainCount != 0 or TestCount != 0:
                        self.Unlink(os.listdir(self.Train), self.Train)
                        self.Unlink(os.listdir(self.Test), self.Test)
                else:
                        print('DataSet directory is ready for procces')
        def Unlink(self, obj, parent):
        #Unlinks pre created symlinks for generating uniqe split
                for y in obj:
                        [os.remove(os.path.join(parent, y, i)) for i in os.listdir(os.path.join(parent, y))]
        def CreatePlaceHolder(self):
        #Fills Place Hodler object for further usage, splites every class with given DivisionParam
                train, test = [], []
                for i in os.listdir(self.DataSet):
                        X = os.listdir(os.path.join(self.DataSet, i))
                        y = np.zeros(len(X))
                        X_train, X_test, _, _ = model_selection.train_test_split(X, y,test_size=self.DivisionParam)
                        X_train = [(os.path.join(self.DataSet, i), i, y) for y in X_train]
                        X_test = [(os.path.join(self.DataSet, i), i, y) for y in X_test]                 
                        train = train + X_train
                        test = test + X_test
                self.PlaceHolder['Train'] = train
                self.PlaceHolder['Test'] = test
        def CreateSym(self):
                #Creates symlinks according to created placeholder object
                self.CheckOut()
                def helper(_iter, parent):
                        if parent == 'Train':
                                commandHolder = self.Train
                        else:
                                 commandHolder = self.Test
                        src = os.path.join( _iter[0], _iter[2])
                        dst = os.path.join( commandHolder, _iter[1], _iter[2])
                        os.symlink(src, dst)
                for i in self.PlaceHolder:
                        print(i)
                        [helper(y, i) for y in self.PlaceHolder[i]]
        def SaveJson(self, path):
                with open(path, 'w') as outfile:
                        json.dump(self.PlaceHolder, outfile)
        def Run(self):
                self.CreatePlaceHolder()
                self.CreateSym()
                print('Train images count: {}\n Test images count: {}'.format(len(self.PlaceHolder['Train']), len(self.PlaceHolder['Test'])))        
      
                
                        
