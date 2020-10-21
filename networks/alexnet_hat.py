import sys
import torch
import numpy as np

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nhid=2000,pdrop1=0.2,pdrop2=0.5,args=0):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.nhid = nhid

        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.fc1=torch.nn.Linear(64*self.smid*self.smid,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(nhid,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)
        self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        print('AlexNetHat')
        print('pdrop1: ',pdrop1)
        print('pdrop2: ',pdrop2)

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gfc1,gfc2=masks
        # Gated
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gfc1,gfc2=masks

        # stack cnn
        # print(gc1.size()) #torch.Size([128, 64, 3, 3])
        # print(gc2.size()) #torch.Size([1, 64])
        # print(gc3.size()) #torch.Size([1, 256])
        # print(self.ec3.weight.size()) #torch.Size([5, 256])
        # print(self.fc1.weight.size()) #torch.Size([2048, 1024])
        # print(self.smid) #2
        # print(self.c2.weight.size()) #torch.Size([1, 128])


        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc1.data.view(-1,1,1).expand((self.ec1.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)

        return None
