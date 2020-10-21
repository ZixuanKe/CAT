import sys
import torch
import numpy as np
import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nhid,args=0):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.ntasks = len(self.taskcla)

        """
        # Config of Sec 2.5 in the paper
        expand_factor = 0.231 # to match num params
        self.N = 5
        self.M = 20     # Large M numbers like this, given our architecture, produce no training
        #"""
        """
        # Config of Sec 2.4 in the paper
        expand_factor = 0.325 # match num params
        self.N = 3
        self.M = 10
        #"""
        #"""
        # Better config found by us
        expand_factor = 0.258 # match num params
        self.N = 3
        self.M = 16
        #"""
        self.L = 2      # our architecture has 5 layers

        self.bestPath = -1 * np.ones((self.ntasks,self.L,self.N),dtype=np.int) #we need to remember this between the tasks

        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2


        #init modules subnets

        self.fc1=torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor*nhid)

        self.fc2=torch.nn.ModuleList()
        self.sizefc2 = int(expand_factor*nhid)

        self.last=torch.nn.ModuleList()

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)


        #declare task columns subnets
        for j in range(self.M):
            self.fc1.append(torch.nn.Linear(ncha*size*size,self.sizefc1))
            self.fc2.append(torch.nn.Linear(self.sizefc1,self.sizefc2))

        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(self.sizefc2,n))


        print('MlpPathNet')
        print('pdrop1: ',pdrop1)
        print('pdrop2: ',pdrop2)


        return

    def forward(self,x,t,P=None):
        if P is None:
            P = self.bestPath[t]
        # P is the genotype path matrix shaped LxN(no.layers x no.permitted modules
        # )
        h=self.drop1(x.view(x.size(0),-1))

        h_pre=self.drop2(self.relu(self.fc1[P[0,0]](h)))
        for j in range(1,self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc1[P[0,j]](h))) #sum activations
        h = h_pre

        h_pre=self.drop2(self.relu(self.fc2[P[1,0]](h)))
        for j in range(1,self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc2[P[1,j]](h))) #sum activations
        h = h_pre

        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

    def unfreeze_path(self,t,Path):
        #freeze modules not in path P and the ones in bestPath paths for the previous tasks
        for i in range(self.M):
            self.unfreeze_module(self.fc1,i,Path[0,:],self.bestPath[0:t,0,:])
            self.unfreeze_module(self.fc2,i,Path[1,:],self.bestPath[0:t,1,:])
        return

    def unfreeze_module(self,layer,i,Path,bestPath):
        if (i in Path) and (i not in bestPath): #if the current module is in the path and not in the bestPath
            utils.set_req_grad(layer[i],True)
        else:
            utils.set_req_grad(layer[i],False)
        return


