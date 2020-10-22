import sys
import torch
import numpy as np

import utils
import math
import torch.nn.functional as F
from torch.autograd import Variable

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,voc_size,weights_matrix,args):
        super(Net,self).__init__()

        self.FILTERS = [3, 4, 5]

        ncha,width,height=inputsize
        self.taskcla=taskcla
        self.WORD_DIM = height
        self.MAX_SENT_LEN = width

        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.filters = args.filters
        self.FILTER_NUM=[args.filters_num] * self.filters

        self.embedding = torch.nn.Embedding(voc_size, self.WORD_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = False # non trainable

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)


        self.mcl = MainContinualLearning(ncha,width,height,self.taskcla,args)
        self.ac = Acessibility(ncha,width,height,self.taskcla,args)

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'

        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1,phase=None,smax=None,args=None):

        if phase==None or args==None:
            raise NotImplementedError
        # Gates
        masks=self.mask(t,s=s,phase=phase,smax=smax,args=args)

        if args.filters>2:
            gc1,gc2,gc3,gfc1,gfc2,gfc3=masks
        elif args.filters>1:
            gc1,gc2,gfc1,gfc2,gfc3=masks
        else:
            gc1,gfc1,gfc2,gfc3=masks

        h = self.drop1(self.embedding(x).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN))

        h1 = F.max_pool1d(self.drop1(F.relu(self.mcl.c1(h))), self.MAX_SENT_LEN - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h1=h1*gc1.view(1,-1,1).expand_as(h1)

        if args.filters>1:
            h2 = F.max_pool1d(self.drop1(F.relu(self.mcl.c2(h))), self.MAX_SENT_LEN - self.FILTERS[1] + 1).view(-1, self.FILTER_NUM[1],1)
            h2=h2*gc2.view(1,-1,1).expand_as(h2)
        if args.filters>2:
            h3 = F.max_pool1d(self.drop2(F.relu(self.mcl.c3(h))), self.MAX_SENT_LEN - self.FILTERS[2] + 1).view(-1, self.FILTER_NUM[2],1)
            h3=h3*gc3.view(1,-1,1).expand_as(h3)

        if args.filters>2:
            h = torch.cat([h1,h2,h3], 1)
            h=h.view(x.size(0),-1)
        elif args.filters>1:
            h = torch.cat([h1,h2], 1)
            h=h.view(x.size(0),-1)
        else:
            h=h1.view(x.size(0),-1)
        h=self.drop2(self.relu(self.mcl.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.mcl.fc2(h)))
        h=h*gfc2.expand_as(h)
        # h=self.drop2(self.relu(self.mcl.fc3(h)))
        # h=h*gfc3.expand_as(h)


        if phase == 'ac':
            y=[]
            for t,i in self.taskcla:
                y.append(self.ac.last[t](h))
            return y,masks

        elif phase == 'mcl':
            y=[]
            for t,i in self.taskcla:
                y.append(self.mcl.last[t](h))
            return y,masks





    def mask(self,t,s=1,phase=None,smax=None,args=None):

        if args==None or phase==None:
            raise NotImplementedError

        if t>0:
            if phase == 'ac':

                ac_gc1=self.gate(s*self.ac.ec1(t))
                gc1=ac_gc1

                if args.filters>1:
                    ac_gc2=self.gate(s*self.ac.ec2(t))
                    gc2=ac_gc2
                if args.filters>2:
                    ac_gc3=self.gate(s*self.ac.ec3(t))
                    gc3=ac_gc3

                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))
                ac_gfc3=self.gate(s*self.ac.efc3(t))

                gfc1=ac_gfc1
                gfc2=ac_gfc2
                gfc3=ac_gfc3

            elif phase == 'mcl':

                ac_gc1=self.gate(smax*self.ac.ec1(t))
                gc1=ac_gc1

                if args.filters>1:
                    ac_gc2=self.gate(smax*self.ac.ec2(t))
                    gc2=ac_gc2
                if args.filters>2:
                    ac_gc3=self.gate(smax*self.ac.ec3(t))
                    gc3=ac_gc3


                ac_gfc1=self.gate(smax*self.ac.efc1(t))
                ac_gfc2=self.gate(smax*self.ac.efc2(t))
                ac_gfc3=self.gate(smax*self.ac.efc3(t))

                gfc1=ac_gfc1
                gfc2=ac_gfc2
                gfc3=ac_gfc3


        elif t==0:
            ac_gc1=self.gate(s*self.ac.ec1(t))
            ac_gc1=torch.ones_like(ac_gc1)
            gc1=ac_gc1

            if args.filters>1:
                ac_gc2=self.gate(s*self.ac.ec2(t))
                ac_gc2=torch.ones_like(ac_gc2)
                gc2=ac_gc2
            if args.filters>2:
                ac_gc3=self.gate(s*self.ac.ec3(t))
                ac_gc3=torch.ones_like(ac_gc3)
                gc3=ac_gc3

            ac_gfc1=self.gate(s*self.ac.efc1(t))
            ac_gfc2=self.gate(s*self.ac.efc2(t))
            ac_gfc3=self.gate(s*self.ac.efc3(t))

            ac_gfc1=torch.ones_like(ac_gfc1)
            ac_gfc2=torch.ones_like(ac_gfc2)
            ac_gfc3=torch.ones_like(ac_gfc3)

            gfc1=ac_gfc1
            gfc2=ac_gfc2
            gfc3=ac_gfc3

        if args.filters>2:
            return [gc1,gc2,gc3,gfc1,gfc2,gfc3]
        elif args.filters>1:
            return [gc1,gc2,gfc1,gfc2,gfc3]
        else:
            return [gc1,gfc1,gfc2,gfc3]



class Acessibility(torch.nn.Module):

    def __init__(self,ncha,width,height,taskcla,args):

        super(Acessibility, self).__init__()
        self.FILTERS = [3, 4, 5]
        self.filters = args.filters
        self.FILTER_NUM=[args.filters_num] * self.filters
        self.WORD_DIM = height
        self.MAX_SENT_LEN = width

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))



        self.ec1=torch.nn.Embedding(len(taskcla),self.FILTER_NUM[0])
        self.ec2=torch.nn.Embedding(len(taskcla),self.FILTER_NUM[0])
        self.ec3=torch.nn.Embedding(len(taskcla),self.FILTER_NUM[0])

        self.efc1=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc3=torch.nn.Embedding(len(taskcla),args.nhid)


class MainContinualLearning(torch.nn.Module):

    def __init__(self,ncha,width,height,taskcla,args):

        super(MainContinualLearning, self).__init__()
        self.WORD_DIM = height
        self.MAX_SENT_LEN = width

        self.FILTERS = [3, 4, 5]
        self.filters = args.filters
        self.FILTER_NUM=[args.filters_num] * self.filters

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)

        if args.filters > 1:
            self.c2 = torch.nn.Conv1d(1, self.FILTER_NUM[1], self.WORD_DIM * self.FILTERS[1], stride=self.WORD_DIM)
        if args.filters > 2:
            self.c3 = torch.nn.Conv1d(1, self.FILTER_NUM[2], self.WORD_DIM * self.FILTERS[2], stride=self.WORD_DIM)

        self.fc1=torch.nn.Linear(self.filters * self.FILTER_NUM[0],args.nhid)
        self.fc2=torch.nn.Linear(args.nhid,args.nhid)
        self.fc3=torch.nn.Linear(args.nhid,args.nhid)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))
