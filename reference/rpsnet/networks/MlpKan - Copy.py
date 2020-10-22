import sys
import torch
import math
import utils
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nlayers=2,nhid=2000,pdrop1=0.2,pdrop2=0.5,args=None):
        super(Net,self).__init__()

        ncha,size,size_height=inputsize
        self.taskcla=taskcla

        self.nlayers=nlayers
        self.relu=torch.nn.ReLU()

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2

        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.nhid=nhid
        self.gate=torch.nn.Sigmoid()

        self.ac = Acessibility(nhid,ncha,size,self.taskcla)
        self.mcl = MainContinualLearning(nhid,ncha,size,self.taskcla)
        self.transfer = TransferLayer(self.taskcla,nhid,ncha,size,args)

        print('MlpTaskKan')


        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return


    # progressive style
    def forward(self,t,x,s=1,phase=None,smax=None,args=None,
                pre_mask=None, pre_task=None,ac_mask=None):
        # Gates
        if phase==None or args==None or self.nlayers>2:
            raise NotImplementedError

        if 'mcl' in phase or 'ac' in phase:
            max_masks=self.mask(t,s=s,phase=phase,smax=smax,args=args)
            gfc1,gfc2=max_masks

            if 'ac-mask' in args.note and phase == 'ac':
                gfc1_ac_mask,gfc2_ac_mask=ac_mask

                gfc1=gfc1*(gfc1_ac_mask.data.clone().detach())
                gfc2=gfc2*(gfc2_ac_mask.data.clone().detach())

                print('gfc1: ',gfc1.round().nonzero().size(0))

            # Gated
            h=self.drop1(x.view(x.size(0),-1))

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



        elif phase == 'transfer':
            gfc1,gfc2=pre_mask
            if 'previous-initial-aan' in args.note:

                h=self.drop1(x.view(x.size(0),-1))

                h=self.drop2(self.relu(self.transfer.previous_aan.fc1(h)))
                h=h*gfc1.expand_as(h)

                h=self.drop2(self.relu(self.transfer.previous_aan.fc2(h)))
                h=h*gfc2.expand_as(h)

                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer[pre_task][t](self.transfer.previous_aan.last[pre_task](h)))

                return y

            else:
                # Gated
                h=self.drop1(x.view(x.size(0),-1))

                h=self.drop2(self.relu(self.mcl.fc1(h)))
                h=h*gfc1.expand_as(h)

                h=self.drop2(self.relu(self.mcl.fc2(h)))
                h=h*gfc2.expand_as(h)

                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer[pre_task][t](self.mcl.last[pre_task](h)))
                return y

        elif phase == 'reference':
            # gfc1,gfc2=self.transfer_mask(t,s=s,phase=phase,smax=smax,args=args)
            if 'premask' in args.note:
                gfc1,gfc2=pre_mask

            if 'random-initial-aan' in args.note:
                h=self.drop1(x.view(x.size(0),-1))
                h=self.drop2(self.relu(self.transfer.fc1(h)))
                if 'premask' in args.note:
                    h=h*gfc1.expand_as(h)
                h=self.drop2(self.relu(self.transfer.fc2(h)))
                if 'premask' in args.note:
                    h=h*gfc2.expand_as(h)

                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer[pre_task][t](self.transfer.last[pre_task](h)))
                return y

            else:
                h=self.drop1(x.view(x.size(0),-1))
                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer_current[t](h))
                return y


    def transfer_mask(self,t,s=1,phase=None,smax=None,args=None):
        if self.nlayers>2 or args==None or phase==None:
            raise NotImplementedError

        gfc1=self.gate(s*self.transfer.efc1(t))
        gfc2=self.gate(s*self.transfer.efc2(t))

        return [gfc1,gfc2]

    def mask(self,t,s=1,phase=None,smax=None,args=None):
        if self.nlayers>2 or args==None or phase==None:
            raise NotImplementedError

        if t>0:
            if phase == 'ac':
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc2=self.gate(s*self.ac.efc2(t))

                gfc1=ac_gfc1
                gfc2=ac_gfc2


            elif phase == 'mcl' or phase == 'transfer':
                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))

                gfc1=mcl_gfc1
                gfc2=mcl_gfc2


        elif t==0:
            if phase == 'ac':
                ac_gfc1=self.gate(s*self.ac.efc1(t))
                ac_gfc1=torch.ones_like(ac_gfc1)

                ac_gfc2=self.gate(s*self.ac.efc2(t))
                ac_gfc2=torch.ones_like(ac_gfc2)

                gfc1=ac_gfc1
                gfc2=ac_gfc2

            elif phase == 'mcl' or phase == 'transfer':

                mcl_gfc1=self.gate(s*self.mcl.efc1(t))
                mcl_gfc2=self.gate(s*self.mcl.efc2(t))

                gfc1=mcl_gfc1
                gfc2=mcl_gfc2

        return [gfc1,gfc2]


    def ac_mask(self,t,phase=None,smax=None,args=None):
        if self.nlayers>2 or args==None or phase==None:
            raise NotImplementedError

        if t>0:
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc2=self.gate(smax*self.ac.efc2(t))

        elif t==0:
            ac_gfc1=self.gate(smax*self.ac.efc1(t))
            ac_gfc1=torch.ones_like(ac_gfc1)

            ac_gfc2=self.gate(smax*self.ac.efc2(t))
            ac_gfc2=torch.ones_like(ac_gfc2)

        return [ac_gfc1,ac_gfc2]

    def get_view_for(self,n,masks):

        gfc1,gfc2=masks
        if n=='mcl.fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.mcl.fc1.weight)
        elif n=='mcl.fc1.bias':
            return gfc1.data.view(-1)
        elif n=='mcl.fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.mcl.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.mcl.fc2.weight)
            return torch.min(post,pre)
        elif n=='mcl.fc2.bias':
            return gfc2.data.view(-1)
        return None


    def ac_get_view_for(self,n,masks):
        gfc1,gfc2=masks

        if n=='ac.fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.ac.fc1.weight)
        elif n=='ac.fc1.bias':
            return gfc1.data.view(-1)
        elif n=='ac.fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.ac.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.ac.fc2.weight)
            return torch.min(post,pre)
        elif n=='ac.fc2.bias':
            return gfc2.data.view(-1)
        return None

class Acessibility(torch.nn.Module):

    def __init__(self,nhid,ncha,size,taskcla):

        super(Acessibility, self).__init__()

        self.efc1=torch.nn.Embedding(len(taskcla),nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),nhid)

        # if 'celeba' in args.experiment:
        #     self.fc1=torch.nn.Linear(ncha*size*size_height,nhid)
        # else:
        #     self.fc1=torch.nn.Linear(ncha*size*size,nhid)

        self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        # self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(nhid,n))

class MainContinualLearning(torch.nn.Module):

    def __init__(self,nhid,ncha,size,taskcla):

        super(MainContinualLearning, self).__init__()

        self.efc1=torch.nn.Embedding(len(taskcla),nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),nhid)

        # if 'celeba' in args.experiment:
        #     self.fc1=torch.nn.Linear(ncha*size*size_height,nhid)
        # else:
        #     self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        # self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)

        self.last=torch.nn.ModuleList()
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(nhid,n))

class TransferLayer(torch.nn.Module):

    def __init__(self,taskcla,nhid,ncha,size,args):

        super(TransferLayer, self).__init__()

        self.efc1=torch.nn.Embedding(len(taskcla),nhid)
        self.efc2=torch.nn.Embedding(len(taskcla),nhid)

        self.transfer=torch.nn.ModuleList()
        for from_t,from_n in taskcla:
            self.transfer_to_n=torch.nn.ModuleList()
            for to_t,to_n in taskcla:
                self.transfer_to_n.append(torch.nn.Linear(from_n,to_n))
            self.transfer.append(self.transfer_to_n)


        if 'random-initial-aan' in args.note:

            self.fc1=torch.nn.Linear(ncha*size*size,nhid)
            self.fc2=torch.nn.Linear(nhid,nhid)

            self.last=torch.nn.ModuleList()
            for t,n in taskcla:
                self.last.append(torch.nn.Linear(nhid,n))

        else:
            self.transfer_current=torch.nn.ModuleList()
            for to_t,to_n in taskcla:
                self.transfer_current.append(torch.nn.Linear(ncha*size*size,to_n))

        if 'previous-initial-aan' in args.note:
            self.previous_aan = None


