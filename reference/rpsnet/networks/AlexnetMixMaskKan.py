import sys
import torch
import numpy as np

import utils
import math

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.ac = Acessibility(ncha,size,self.taskcla)
        self.mcl = MainContinualLearning(ncha,size,self.taskcla)
        # self.last_layers = LastLayer(self.taskcla)


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
        max_masks=self.mask(t,s=s,phase=phase,smax=smax,args=args)
        gc1,gc2,gc3,gfc1,gfc2=max_masks
        # Gated
        h=self.maxpool(self.drop1(self.relu(self.mcl.c1(x))))
        # print(h.size())
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.mcl.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.mcl.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.mcl.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.mcl.fc2(h)))
        h=h*gfc2.expand_as(h)

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


    def mask(self,t,s=1,phase=None,smax=None,args=None):

        if args==None or phase==None:
            raise NotImplementedError

        if t>0:
            if phase == 'ac':

                ac_gc1=self.gate(s*self.ac.ec1(t))
                ac_gc2=self.gate(s*self.ac.ec2(t))
                ac_gc3=self.gate(s*self.ac.ec3(t))
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))

                gc1=ac_gc1
                gc2=ac_gc2
                gc3=ac_gc3
                gfc1=ac_gfc1
                gfc2=ac_gfc2

            elif phase == 'mcl':

                ac_gc1=self.gate(smax*self.ac.ec1(t))
                ac_gc2=self.gate(smax*self.ac.ec2(t))
                ac_gc3=self.gate(smax*self.ac.ec3(t))
                ac_gfc1=self.gate(smax*self.ac.efc1(t))
                ac_gfc2=self.gate(smax*self.ac.efc2(t))


                mcl_gc1=self.gate(s*self.mcl.ec1(t))
                mcl_gc2=self.gate(s*self.mcl.ec2(t))
                mcl_gc3=self.gate(s*self.mcl.ec3(t))
                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))

                gc1=torch.max(mcl_gc1,ac_gc1)
                gc2=torch.max(mcl_gc2,ac_gc2)
                gc3=torch.max(mcl_gc3,ac_gc3)
                gfc1=torch.max(mcl_gfc1,ac_gfc1)
                gfc2=torch.max(mcl_gfc2,ac_gfc2)


                # gc1=mcl_gc1
                # gc2=mcl_gc2
                # gc3=mcl_gc3
                # gfc1=mcl_gfc1
                # gfc2=mcl_gfc2



        elif t==0:
            if phase == 'ac':
                ac_gc1=self.gate(s*self.ac.ec1(t))
                ac_gc2=self.gate(s*self.ac.ec2(t))
                ac_gc3=self.gate(s*self.ac.ec3(t))
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))

                ac_gc1=torch.ones_like(ac_gc1)
                ac_gc2=torch.ones_like(ac_gc2)
                ac_gc3=torch.ones_like(ac_gc3)
                ac_gfc1=torch.ones_like(ac_gfc1)
                ac_gfc2=torch.ones_like(ac_gfc2)

                gc1=ac_gc1
                gc2=ac_gc2
                gc3=ac_gc3
                gfc1=ac_gfc1
                gfc2=ac_gfc2

            elif phase == 'mcl':

                mcl_gc1=self.gate(s*self.mcl.ec1(t))
                mcl_gc2=self.gate(s*self.mcl.ec2(t))
                mcl_gc3=self.gate(s*self.mcl.ec3(t))
                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))

                gc1=mcl_gc1
                gc2=mcl_gc2
                gc3=mcl_gc3
                gfc1=mcl_gfc1
                gfc2=mcl_gfc2

        return [gc1,gc2,gc3,gfc1,gfc2]

    def ac_mask(self,t,phase=None,smax=None,args=None):
        if args==None or phase==None:
            raise NotImplementedError


        if t>0:
            ac_gc1=self.gate(smax*self.ac.ec1(t))
            ac_gc2=self.gate(smax*self.ac.ec2(t))
            ac_gc3=self.gate(smax*self.ac.ec3(t))
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc2=self.gate(smax*self.ac.efc2(t))

            gc1=ac_gc1
            gc2=ac_gc2
            gc3=ac_gc3
            gfc1=ac_gfc1
            gfc2=ac_gfc2


        elif t==0:
            ac_gc1=self.gate(smax*self.ac.ec1(t))
            ac_gc2=self.gate(smax*self.ac.ec2(t))
            ac_gc3=self.gate(smax*self.ac.ec3(t))
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc2=self.gate(smax*self.ac.efc2(t))

            ac_gc1=torch.ones_like(ac_gc1)
            ac_gc2=torch.ones_like(ac_gc2)
            ac_gc3=torch.ones_like(ac_gc3)
            ac_gfc1=torch.ones_like(ac_gfc1)
            ac_gfc2=torch.ones_like(ac_gfc2)

            gc1=ac_gc1
            gc2=ac_gc2
            gc3=ac_gc3
            gfc1=ac_gfc1
            gfc2=ac_gfc2

        return [gc1,gc2,gc3,gfc1,gfc2]


    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='mcl.fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.mcl.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.mcl.ec3.weight.size(1),self.mcl.smid,self.mcl.smid)).contiguous().view(1,-1).expand_as(self.mcl.fc1.weight)
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
            return gc1.data.view(-1,1,1,1).expand_as(self.mcl.c1.weight)
        elif n=='mcl.c1.bias':
            return gc1.data.view(-1)
        elif n=='mcl.c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.mcl.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.mcl.c2.weight)
            return torch.min(post,pre)
        elif n=='mcl.c2.bias':
            return gc2.data.view(-1)
        elif n=='mcl.c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.mcl.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.mcl.c3.weight)
            return torch.min(post,pre)
        elif n=='mcl.c3.bias':
            return gc3.data.view(-1)
        return None



    def ac_get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='ac.fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.ac.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.ac.ec3.weight.size(1),self.ac.smid,self.ac.smid)).contiguous().view(1,-1).expand_as(self.ac.fc1.weight)
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
            return gc1.data.view(-1,1,1,1).expand_as(self.ac.c1.weight)
        elif n=='ac.c1.bias':
            return gc1.data.view(-1)
        elif n=='ac.c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.ac.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.ac.c2.weight)
            return torch.min(post,pre)
        elif n=='ac.c2.bias':
            return gc2.data.view(-1)
        elif n=='ac.c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.ac.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.ac.c3.weight)
            return torch.min(post,pre)
        elif n=='ac.c3.bias':
            return gc3.data.view(-1)
        return None


class Acessibility(torch.nn.Module):

    def __init__(self,ncha,size,taskcla):

        super(Acessibility, self).__init__()

        self.ec1=torch.nn.Embedding(len(taskcla),64)
        self.ec2=torch.nn.Embedding(len(taskcla),128)
        self.ec3=torch.nn.Embedding(len(taskcla),256)
        self.efc1=torch.nn.Embedding(len(taskcla),2048)
        self.efc2=torch.nn.Embedding(len(taskcla),2048)

        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s

        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(2048,n))


class MainContinualLearning(torch.nn.Module):

    def __init__(self,ncha,size,taskcla):

        super(MainContinualLearning, self).__init__()

        self.ec1=torch.nn.Embedding(len(taskcla),64)
        self.ec2=torch.nn.Embedding(len(taskcla),128)
        self.ec3=torch.nn.Embedding(len(taskcla),256)
        self.efc1=torch.nn.Embedding(len(taskcla),2048)
        self.efc2=torch.nn.Embedding(len(taskcla),2048)


        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s

        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(2048,n))


# class LastLayer(torch.nn.Module):
#
#     def __init__(self,taskcla):
#
#         super(LastLayer, self).__init__()
#
#         self.last=torch.nn.ModuleList()
#         for t,n in taskcla:
#             self.last.append(torch.nn.Linear(2048,n))