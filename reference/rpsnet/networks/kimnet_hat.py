import sys
import torch
import numpy as np

import utils
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,voc_size,weights_matrix,args):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[args.filter_num] *3
        self.WORD_DIM = 300
        self.MAX_SENT_LEN = 240

        self.filters = 1
        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.embedding = torch.nn.Embedding(voc_size, self.WORD_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = False # non trainable

        self.relu=torch.nn.ReLU()

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)

        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.fc1=torch.nn.Linear(self.filters * self.FILTER_NUM[0],args.nhid)
        self.fc2=torch.nn.Linear(args.nhid,args.nhid)
        self.fc3=torch.nn.Linear(args.nhid,args.nhid)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[0])
        self.efc1=torch.nn.Embedding(len(self.taskcla),args.nhid)
        self.efc2=torch.nn.Embedding(len(self.taskcla),args.nhid)
        self.efc3=torch.nn.Embedding(len(self.taskcla),args.nhid)

        print('KimNetHAT')

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gfc1,gfc2,gfc3=masks

        h = self.drop1(self.embedding(x)).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)

        h1 = F.max_pool1d(self.drop1(F.relu(self.c1(h))), self.MAX_SENT_LEN - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h1=h1*gc1.view(1,-1,1).expand_as(h1)

        h=h1.view(x.size(0),-1)

        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        #
        # h=self.drop2(self.relu(self.fc3(h)))
        # h=h*gfc3.expand_as(h)


        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))

        return y,masks


    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        gfc3=self.gate(s*self.efc3(t))

        return [gc1,gfc1,gfc2,gfc3]


    def get_view_for(self,n,masks):
        gc1,gfc1,gfc2,gfc3=masks

        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc1.data.view(-1,1).expand((self.ec1.weight.size(1),self.filters)).contiguous().view(1,-1).expand_as(self.fc1.weight)
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
            return gc1.data.view(-1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif n=='fc3.bias':
            return gfc3.data.view(-1)
        return None
