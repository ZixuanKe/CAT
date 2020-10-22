import sys,time
import numpy as np
import torch

import utils

########################################################################################################################

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=56,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.smax=float(params[1])

        self.mask_pre=None
        self.mask_back=None

        return

    def _get_optimizer(self,lr=None,phase=None):
        # if phase==None:
            # raise NotImplementedError
        if lr is None: lr=self.lr
        if phase=='ac':
            # return torch.optim.SGD(list(self.model.ac.parameters())+list(self.model.last_layers.parameters()),lr=lr)
            return torch.optim.SGD(list(self.model.ac.parameters()),lr=lr)

        elif phase=='mcl':
            # return torch.optim.SGD(list(self.model.mcl.parameters())+list(self.model.last_layers.parameters()),lr=lr)
            return torch.optim.SGD(list(self.model.mcl.parameters()),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid,phase,args,
              similarity=None,history_mask_back=None):

        self.similarity=similarity
        self.history_mask_back=history_mask_back

        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr,phase)
        # Loop epochs
        try:
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
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)

        if phase=='mcl':
            # Activations mask
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            mask=self.model.mask(task,s=self.smax,phase=phase,smax=self.smax,args=args)
            for i in range(len(mask)):
                mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
            if t==0:
                self.mask_pre=mask
            else:
                for i in range(len(self.mask_pre)):
                    self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

            # Weights mask
            self.mask_back={}
            for n,_ in self.model.named_parameters():
                vals=self.model.get_view_for(n,self.mask_pre)
                if vals is not None:
                    self.mask_back[n]=1-vals

        return

    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6,phase=None,args=None):
        self.model.train()
        if phase==None or args==None:
            raise NotImplementedError

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
                loss,_=self.criterion(output,targets,masks)


            # Backward
            self.optimizer.zero_grad()
            loss.backward()



            if phase == 'mcl':
                # Restrict layer gradients in backprop
                if t>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back:
                            ac_mask=self.model.ac_mask(task,phase=phase,smax=self.smax,args=args)
                            ac_vals=self.model.ac_get_view_for(n.replace('mcl','ac'),ac_mask)

                            if 'similar' in args.note:
                                if self.similarity is not None:
                                    if 'alter' in args.note:
                                        if t % 2 !=0:
                                            p.grad.data*=(torch.max(self.mask_back[n],ac_vals*1))
                                        elif t % 2 == 0:
                                            p.grad.data*=(torch.max(self.mask_back[n],ac_vals*0))

                                    elif 'stack' in args.note:
                                        if t > 9:
                                            p.grad.data*=(torch.max(self.mask_back[n],ac_vals*1))
                                        elif y <=9:
                                            p.grad.data*=(torch.max(self.mask_back[n],ac_vals*0))

                                else:
                                    p.grad.data*=(torch.max(self.mask_back[n],ac_vals))

                            elif 'similar' not in args.note:
                                p.grad.data*=(torch.max(self.mask_back[n],ac_vals*float(args.note.split(',')[0])))

                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    if n.startswith('mcl.e'):
                        num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den

            elif phase == 'ac':
                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    # print('n: ',n)
                    if n.startswith('ac.e'):
                        num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            if phase == 'mcl':
                # Constrain embeddings
                for n,p in self.model.named_parameters():
                    if n.startswith('mcl.e'):
                        p.data=torch.clamp(p.data,-thres_emb,thres_emb)


            elif phase == 'ac':
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
                loss,_=self.criterion(output,targets,masks)


            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)
            # total_reg+=reg.data.cpu().numpy().item()*len(b)

        # print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num


    def criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

    def ac_criterion(self,outputs,targets,masks):
        reg=0
        count=0
        for m in masks:
            reg+=m.sum()
            count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg
        # return self.ce(outputs,targets),None


########################################################################################################################
