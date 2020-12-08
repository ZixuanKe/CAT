import sys
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nhid=2000,args=0):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.nhid = nhid

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        pdrop1 = args.pdrop1
        pdrop2 = args.pdrop2


        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.fc1=torch.nn.Linear(64*s*s,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(nhid,n))

        print('CNN')

        return

    def forward(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y
