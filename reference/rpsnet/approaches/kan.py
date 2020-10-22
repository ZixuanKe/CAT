import sys,time
import numpy as np
import torch

import utils
from sklearn.metrics import precision_recall_fscore_support

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.model=model

        self.nepochs=nepochs
        if args.sbatch == 0:
            self.sbatch=sbatch
        else:
            self.sbatch=args.sbatch

        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.smax=float(params[1])

        return


    def _get_optimizer(self,lr=None,phase=None):
        if lr is None: lr=self.lr
        if phase=='ac':
            # return torch.optim.SGD(list(self.model.ac.parameters())+list(self.model.last_layers.parameters()),lr=lr)
            return torch.optim.SGD(list(self.model.ac.parameters()),lr=lr)

        elif phase=='mcl':
            # return torch.optim.SGD(list(self.model.mcl.parameters())+list(self.model.last_layers.parameters()),lr=lr)
            return torch.optim.SGD(list(self.model.mcl.parameters()),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid,phase,args): #N-CL
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr,phase)

        if phase == 'mcl':
            print('before: ',self.model.mcl.fc1.weight)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            print('before: ',self.model.mask(task,phase=phase,smax=self.smax,args=args)[0])

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain,phase=phase,args=args)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain,phase=phase,args=args)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid,phase=phase,args=args)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr,phase)
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        if phase == 'mcl':
            print('after: ',self.model.mcl.fc1.weight)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            print('after: ',self.model.mask(task,phase=phase,smax=self.smax,args=args)[0])
        return

    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6,phase=None,args=None):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)

            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            s=(self.smax-1/self.smax)*i/len(r)+1/self.smax

            # Forward
            outputs,masks=self.model.forward(task,images,s=s,phase=phase,smax=self.smax,args=args)
            output=outputs[t]

            if phase == 'ac':
                loss,_=self.ac_criterion(output,targets,masks)
            elif phase == 'mcl':
                loss=self.criterion(output,targets)


            # Backward
            self.optimizer.zero_grad()
            loss.backward()


            if phase == 'ac':
                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    # print('n: ',n)
                    if n.startswith('ac.e') and (p.grad is not None):
                        num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            if phase == 'ac':
                # Constrain embeddings
                for n,p in self.model.named_parameters():
                    if n.startswith('ac.e'):
                        p.data=torch.clamp(p.data,-thres_emb,thres_emb)

        return

    def eval(self,t,x,y,phase=None,args=None):
        total_loss=0
        total_acc=0
        total_num=0

        self.model.eval()

        if phase==None or args==None:
            raise NotImplementedError

        total_reg=0

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            outputs,masks=self.model.forward(task,images,s=self.smax,phase=phase,smax=self.smax,args=args)
            output=outputs[t]

            if phase == 'ac':
                loss,_=self.ac_criterion(output,targets,masks)
            elif phase == 'mcl':
                loss=self.criterion(output,targets)


            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)
            # total_reg+=reg.data.cpu().numpy().item()*len(b)

        # print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num

    def ac_criterion(self,outputs,targets,masks):
        return self.criterion(outputs,targets),None

    # def ac_criterion(self,outputs,targets,masks):
    #     reg=0
    #     count=0
    #     for m in masks:
    #         reg+=m.sum()
    #         count+=np.prod(m.size()).item()
    #     reg/=count
    #     return self.criterion(outputs,targets)+self.lamb*reg,reg
        # return self.ce(outputs,targets),None