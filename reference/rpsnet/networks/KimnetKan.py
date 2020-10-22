import sys
import torch
import numpy as np

import utils
import math
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

        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.filters = 1


        self.embedding = torch.nn.Embedding(voc_size, self.WORD_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = False # non trainable
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.mcl = MainContinualLearning(ncha,size,self.taskcla,args)
        self.ac = Acessibility(ncha,size,self.taskcla,args)

        self.gate=torch.nn.Sigmoid()
        print('KimNetKan')
        return

    def forward(self,t,x,s=1,phase=None,smax=None,args=None):

        if phase==None or args==None:
            raise NotImplementedError
        # Gates
        max_masks=self.mask(t,s=s,phase=phase,smax=smax,args=args)
        gc1,gfc1,gfc2,gfc3=max_masks

        # Gated
        h = self.drop1(self.embedding(x)).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)

        h1 = F.max_pool1d(self.drop1(F.relu(self.mcl.c1(h))), self.MAX_SENT_LEN - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)

        print('h1: ',h1.size())
        h1=h1*gc1.view(1,-1,1).expand_as(h1)

        h=h1.view(x.size(0),-1)

        h=self.drop2(self.relu(self.mcl.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.mcl.fc2(h)))
        h=h*gfc2.expand_as(h)
        #
        # h=self.drop2(self.relu(self.mcl.fc3(h)))
        # h=h*gfc3.expand_as(h)


        if phase == 'ac':
            y=[]
            for t,i in self.taskcla:
                y.append(self.ac.last[t](h))
            return y,max_masks

        elif phase == 'mcl':
            y=[]
            for t,i in self.taskcla:
                y.append(self.mcl.last[t](h))
            return y,max_masks


    def get_view_for(self,n,masks):
        # print(n)
        gc1,gfc1,gfc2,gfc3=masks

        if n=='mcl.fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.mcl.fc1.weight)
            pre=gc1.data.view(-1,1).expand((self.mcl.ec1.weight.size(1),self.filters)).contiguous().view(1,-1).expand_as(self.mcl.fc1.weight)
            return torch.min(post,pre)
        elif n=='mcl.fc1.bias':
            return gfc1.data.view(-1)
        elif n=='mcl.fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.mcl.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.mcl.fc2.weight)
            return torch.min(post,pre)
        elif n=='mcl.fc2.bias':
            return gfc2.data.view(-1)

        elif n=='mcl.c1.weight':
            return gc1.data.view(-1,1,1).expand_as(self.mcl.c1.weight)
        elif n=='mcl.c1.bias':
            return gc1.data.view(-1)

        elif n=='mcl.fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.mcl.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.mcl.fc3.weight)
            return torch.min(post,pre)
        elif n=='mcl.fc3.bias':
            return gfc3.data.view(-1)



        return None



    def ac_get_view_for(self,n,masks):
        gc1,gfc1,gfc2,gfc3=masks

        if n=='ac.fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.ac.fc1.weight)
            pre=gc1.data.view(-1,1).expand((self.ac.ec1.weight.size(1),self.filters)).contiguous().view(1,-1).expand_as(self.ac.fc1.weight)
            return torch.min(post,pre)
        elif n=='ac.fc1.bias':
            return gfc1.data.view(-1)
        elif n=='ac.fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.ac.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.ac.fc2.weight)
            return torch.min(post,pre)
        elif n=='ac.fc2.bias':
            return gfc2.data.view(-1)
        elif n=='ac.c1.weight':
            return gc1.data.view(-1,1,1).expand_as(self.ac.c1.weight)
        elif n=='ac.c1.bias':
            return gc1.data.view(-1)

        elif n=='ac.fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.ac.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.ac.fc3.weight)
            return torch.min(post,pre)
        elif n=='ac.fc3.bias':
            return gfc3.data.view(-1)



        return None




    def mask(self,t,s=1,phase=None,smax=None,args=None):

        if args==None or phase==None:
            raise NotImplementedError

        if t>0:
            if phase == 'ac':

                ac_gc1=self.gate(s*self.ac.ec1(t))
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))
                ac_gfc3=self.gate(s*self.ac.efc3(t))

                gc1=ac_gc1
                gfc1=ac_gfc1
                gfc2=ac_gfc2
                gfc3=ac_gfc3

            elif phase == 'mcl':

                ac_gc1=self.gate(smax*self.ac.ec1(t))
                ac_gfc1=self.gate(smax*self.ac.efc1(t))
                ac_gfc2=self.gate(smax*self.ac.efc2(t))
                ac_gfc3=self.gate(smax*self.ac.efc3(t))

                mcl_gc1=self.gate(s*self.mcl.ec1(t))
                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))
                mcl_gfc3=self.gate(s*self.mcl.efc3(t))

                if 'max' in args.note:
                    gc1=torch.max(mcl_gc1,ac_gc1)
                    gfc1=torch.max(mcl_gfc1,ac_gfc1)
                    gfc2=torch.max(mcl_gfc2,ac_gfc2)
                    gfc3=torch.max(mcl_gfc3,ac_gfc3)

                elif 'norm' in args.note:
                    gc1=mcl_gc1
                    gfc1=mcl_gfc1
                    gfc2=mcl_gfc2
                    gfc3=mcl_gfc3


        elif t==0:
            if phase == 'ac':
                ac_gc1=self.gate(s*self.ac.ec1(t))
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))
                ac_gfc3=self.gate(s*self.ac.efc3(t))


                ac_gc1=torch.ones_like(ac_gc1)
                ac_gfc1=torch.ones_like(ac_gfc1)
                ac_gfc2=torch.ones_like(ac_gfc2)
                ac_gfc3=torch.ones_like(ac_gfc3)


                gc1=ac_gc1
                gfc1=ac_gfc1
                gfc2=ac_gfc2
                gfc3=ac_gfc3

            elif phase == 'mcl':

                mcl_gc1=self.gate(s*self.mcl.ec1(t))
                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))
                mcl_gfc3=self.gate(s*self.mcl.efc3(t))

                gc1=mcl_gc1
                gfc1=mcl_gfc1
                gfc2=mcl_gfc2
                gfc3=mcl_gfc3


        return [gc1,gfc1,gfc2,gfc3]

    def ac_mask(self,t,phase=None,smax=None,args=None):
        if args==None or phase==None:
            raise NotImplementedError


        if t>0:
            ac_gc1=self.gate(smax*self.ac.ec1(t))
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc2=self.gate(smax*self.ac.efc2(t))
            ac_gfc3=self.gate(smax*self.ac.efc3(t))


            gc1=ac_gc1
            gfc1=ac_gfc1
            gfc2=ac_gfc2
            gfc3=ac_gfc3


        elif t==0:
            ac_gc1=self.gate(smax*self.ac.ec1(t))
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc2=self.gate(smax*self.ac.efc2(t))
            ac_gfc3=self.gate(smax*self.ac.efc3(t))

            ac_gc1=torch.ones_like(ac_gc1)
            ac_gfc1=torch.ones_like(ac_gfc1)
            ac_gfc2=torch.ones_like(ac_gfc2)
            ac_gfc3=torch.ones_like(ac_gfc3)


            gc1=ac_gc1
            gfc1=ac_gfc1
            gfc2=ac_gfc2
            gfc3=ac_gfc3

        return [gc1,gfc1,gfc2,gfc3]


class Acessibility(torch.nn.Module):

    def __init__(self,ncha,size,taskcla,args):

        super(Acessibility, self).__init__()
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[args.filter_num] *3
        self.WORD_DIM = 300
        self.MAX_SENT_LEN = 240
        self.filters = 1

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)

        self.fc1=torch.nn.Linear(self.filters * self.FILTER_NUM[0],args.nhid)
        self.fc2=torch.nn.Linear(args.nhid,args.nhid)
        self.fc3=torch.nn.Linear(args.nhid,args.nhid)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))

        self.ec1=torch.nn.Embedding(len(taskcla),self.FILTER_NUM[0])
        self.efc1=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc3=torch.nn.Embedding(len(taskcla),args.nhid)


class MainContinualLearning(torch.nn.Module):

    def __init__(self,ncha,size,taskcla,args):

        super(MainContinualLearning, self).__init__()
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[args.filter_num] *3
        self.WORD_DIM = 300
        self.MAX_SENT_LEN = 240
        self.filters = 1

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)

        self.fc1=torch.nn.Linear(self.filters * self.FILTER_NUM[0],args.nhid)
        self.fc2=torch.nn.Linear(args.nhid,args.nhid)
        self.fc3=torch.nn.Linear(args.nhid,args.nhid)


        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))

        self.ec1=torch.nn.Embedding(len(taskcla),self.FILTER_NUM[0])
        self.efc1=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),args.nhid)
        self.efc3=torch.nn.Embedding(len(taskcla),args.nhid)
