import sys
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nhid,args=0):


        super(Net,self).__init__()

        ncha,size,size_height=inputsize
        self.taskcla=taskcla
        # nhid = 800
        # nhid = 2000
        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        # if 'celeba' in args.experiment:
        #     self.fc1=torch.nn.Linear(ncha*size*size_height,nhid)
        # else:
        #     self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)
        # self.fc3=torch.nn.Linear(nhid,nhid)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(nhid,n))

        print('MlpNet')
        print('pdrop1: ',pdrop1)
        print('pdrop2: ',pdrop2)

        return

    def forward(self,x):
        # print('x: ',x.size())

        # h=x.view(x.size(0),-1)
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        # h=self.drop2(self.relu(self.fc3(h)))

        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y
