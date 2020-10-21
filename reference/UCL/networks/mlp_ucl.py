import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianLinear

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, unitN = 400, split = False, notMNIST=False, args=0):
        super().__init__()

        ncha,size,size_height=inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla=taskcla
        self.split = split
        self.experiment = 'xxx'
        self.ncha = ncha
        self.size = size
        self.size_height = size_height

        if 'celeba' in args.experiment:
            self.experiment = 'celeba'
            self.fc1=BayesianLinear(ncha*size*size_height, unitN,ratio=ratio)
        else:
            self.fc1=BayesianLinear(ncha*size*size, unitN,ratio=ratio)

        self.fc2 = BayesianLinear(unitN, unitN,ratio=ratio)
        
        if notMNIST:
            self.fc3=BayesianLinear(unitN,unitN,ratio=ratio)
            self.fc4=BayesianLinear(unitN,unitN,ratio=ratio)
        self.last=torch.nn.ModuleList()
        
        if split:
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))
        
        else:
            self.fc3 = BayesianLinear(unitN, taskcla[0][1],ratio=ratio)
                
        print('MLP-UCL')

    def forward(self, x, sample=False):
        if 'celeba' in self.experiment:
            x = x.view(-1, self.ncha*self.size*self.size_height)
        else:
            x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        if self.notMNIST:
            x=F.relu(self.fc3(x, sample))
            x=F.relu(self.fc4(x, sample))
        
        if self.split:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
            
        else:
            x = self.fc3(x, sample)
            y = F.log_softmax(x, dim=1)
        
        return y

