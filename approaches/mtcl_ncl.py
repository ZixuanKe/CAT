import sys,time
import numpy as np
import torch

import utils
from copy import deepcopy
from itertools import zip_longest

########################################################################################################################

class Appr(object):

    def __init__(self,model,args,lr_min=1e-4,lr_factor=3,clipgrad=10000,lamb=0.75,smax=400):
        self.model=model

        self.nepochs=args.nepochs
        self.sbatch=args.sbatch
        self.lr=args.lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=args.lr_patience
        self.clipgrad=clipgrad
        self.args = args
        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.nepochs_kt=args.nepochs_kt
        self.lr_kt=args.lr_kt
        self.lr_patience_kt=args.lr_patience_kt


        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.smax=float(params[1])

        self.mask_pre=None
        self.mask_back=None

        self.transfer_initial_model = deepcopy(model.transfer)


        return

    def _get_optimizer(self,lr=None,phase=None,args=None):
        # if phase==None:
            # raise NotImplementedError
        if lr is None: lr=self.lr

        elif phase=='mcl' and ('pipeline' in args.loss_type or 'no_attention' in args.loss_type):
            return torch.optim.SGD(list(self.model.mcl.parameters()),lr=lr)

        elif phase=='mcl' and 'joint' in args.loss_type:
            return torch.optim.SGD(list(self.model.kt.parameters())+list(self.model.mcl.parameters()),lr=lr)

        if phase=='kt':
            return torch.optim.SGD(list(self.model.kt.parameters())+list(self.model.mcl.parameters()),lr=lr)

        elif  phase=='transfer':
            return torch.optim.SGD(list(self.model.transfer.parameters()),lr=lr)

        elif  phase=='reference':
            return torch.optim.SGD(list(self.model.transfer.parameters()),lr=lr)



    def train(self,t,xtrain,ytrain,xvalid,yvalid,phase,args,
              pre_mask=None,pre_task=None,
              similarity=None,history_mask_back=None,
              history_mask_pre=None,check_federated=None):

        self.model.transfer=deepcopy(self.transfer_initial_model) # Restart transfer network: isolate

        best_loss=np.inf
        best_model=utils.get_model(self.model)


        if phase=='kt':
            lr=self.lr_kt
            patience=self.lr_patience_kt
            nepochs=self.nepochs_kt
        elif phase=='mcl' or phase=='transfer' or phase=='reference':
            lr=self.lr
            patience=self.lr_patience
            nepochs=self.nepochs


        self.optimizer=self._get_optimizer(lr,phase,args)
        print('similarity: ',similarity)

        try:
            for e in range(nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain,phase=phase,pre_mask=pre_mask,
                                 pre_task=pre_task,similarity=similarity,history_mask_back=history_mask_back,
                                 history_mask_pre=history_mask_pre,check_federated=check_federated)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain,phase=phase,pre_mask=pre_mask,
                                               pre_task=pre_task,similarity=similarity,
                                               history_mask_pre=history_mask_pre,check_federated=check_federated)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid,phase=phase,pre_mask=pre_mask,
                                               pre_task=pre_task,similarity=similarity,
                                               history_mask_pre=history_mask_pre,check_federated=check_federated)
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
                        self.optimizer=self._get_optimizer(lr,phase,args)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)


        if phase=='mcl':
            # Activations mask
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            mask=self.model.mask(task,s=self.smax)


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

    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6,phase=None,
                    pre_mask=None, pre_task=None,
                    similarity=None,history_mask_back=None,
                    history_mask_pre=None,check_federated=None):
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

            if phase == 'mcl':
                outputs,masks,outputs_attn=self.model.forward(task,images,s=s,phase=phase,
                                                             similarity=similarity,history_mask_pre=history_mask_pre,
                                                              check_federated=check_federated)
                output=outputs[t]

                if outputs_attn is None:
                    loss=self.criterion(output,targets,masks)
                else:
                    output_attn=outputs_attn[t]
                    loss=self.joint_criterion(output,targets,masks,output_attn)


            elif phase == 'transfer' or phase == 'reference':

                outputs=self.model.forward(task,images,s=s,phase=phase,
                                                 pre_mask=pre_mask, pre_task=pre_task)
                output=outputs[t]
                loss=self.transfer_criterion(output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()



            if phase == 'mcl':
                # Restrict layer gradients in backprop
                if t>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back and p.grad is not None:
                            Tsim_mask=self.model.Tsim_mask(task,history_mask_pre=history_mask_pre,similarity=similarity)
                            Tsim_vals=self.model.get_view_for(n,Tsim_mask).clone()
                            p.grad.data*=torch.max(self.mask_back[n],Tsim_vals)


                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    if n.startswith('mcl.e') and p.grad is not None:
                        num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den



            elif phase == 'reference':
                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    if n.startswith('transfer.e')  and p.grad is not None:
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


            elif phase == 'reference':
                # Constrain embeddings
                for n,p in self.model.named_parameters():
                    if n.startswith('transfer.e'):
                        p.data=torch.clamp(p.data,-thres_emb,thres_emb)
        return



    def eval(self,t,x,y,phase=None,
            pre_mask=None, pre_task=None,similarity=None,
             history_mask_pre=None,check_federated=None
             ):
        total_att_loss=0
        total_att_acc=0

        total_mask_loss=0
        total_mask_acc=0

        total_num=0
        self.model.eval()

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

            if phase == 'mcl':
                outputs,masks,outputs_attn=self.model.forward(task,images,s=self.smax,phase=phase,
                                                             similarity=similarity,
                                                             history_mask_pre=history_mask_pre,
                                                             check_federated=check_federated)
                output=outputs[t]

                if outputs_attn is None:
                    loss=self.criterion(output,targets,masks)
                else:
                    output_attn=outputs_attn[t]
                    loss=self.joint_criterion(output,targets,masks,output_attn)

            elif phase == 'transfer' or phase == 'reference':
                outputs=self.model.forward(task,images,s=self.smax,phase=phase,
                                                 pre_mask=pre_mask, pre_task=pre_task)
                output=outputs[t]
                loss=self.transfer_criterion(output,targets)


            # if phase=='mcl' and (similarity is not None and t<len(similarity) and np.count_nonzero(similarity[:t])>1 and similarity[t]==1):

            if phase=='mcl' and 'no_attention' not in self.args.loss_type and outputs_attn is not None:
                _,att_pred=output_attn.max(1)
                _,mask_pred=output.max(1)

                mask_hits=(mask_pred==targets).float()
                att_hits=(att_pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=mask_hits.sum().data.cpu().numpy().item()

                # Log
                total_att_loss+=loss.data.cpu().numpy().item()*len(b)
                total_att_acc+=att_hits.sum().data.cpu().numpy().item()


            else:
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=hits.sum().data.cpu().numpy().item()


            total_num+=len(b)

        if 'all-one' in self.args.similarity_detection:
                total_loss = total_att_loss
                total_acc = total_att_acc

        elif phase=='mcl' and 'no_attention' not in self.args.loss_type:
            if total_att_acc > total_mask_acc:
                total_loss = total_att_loss
                total_acc = total_att_acc
            else:
                total_loss = total_mask_loss
                total_acc = total_mask_acc

        else:
                total_loss = total_mask_loss
                total_acc = total_mask_acc

        return total_loss/total_num,total_acc/total_num


    def test(self,t,x,y,phase=None,
            pre_mask=None, pre_task=None,similarity=None,
             history_mask_pre=None,check_federated=None,xvalid=None,yvalid=None
             ):
        choose_att = False
        total_att_loss=0
        total_att_acc=0

        total_mask_loss=0
        total_mask_acc=0

        total_num=0

        self.model.eval()

        r=np.arange(xvalid.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(xvalid[b],volatile=True)
            targets=torch.autograd.Variable(yvalid[b],volatile=True)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward

            if phase == 'mcl':
                outputs,masks,outputs_attn=self.model.forward(task,images,s=self.smax,phase=phase,
                                                             similarity=similarity,
                                                             history_mask_pre=history_mask_pre,
                                                             check_federated=check_federated)
                output=outputs[t]

                if outputs_attn is None:
                    loss=self.criterion(output,targets,masks)
                else:
                    output_attn=outputs_attn[t]
                    loss=self.joint_criterion(output,targets,masks,output_attn)

            elif phase == 'transfer' or phase == 'reference':
                outputs=self.model.forward(task,images,s=self.smax,phase=phase,
                                                 pre_mask=pre_mask, pre_task=pre_task)
                output=outputs[t]
                loss=self.transfer_criterion(output,targets)


            # if phase=='mcl' and (similarity is not None and t<len(similarity) and np.count_nonzero(similarity[:t])>1 and similarity[t]==1):

            if phase=='mcl' and 'no_attention' not in self.args.loss_type and outputs_attn is not None:
                _,att_pred=output_attn.max(1)
                _,mask_pred=output.max(1)

                mask_hits=(mask_pred==targets).float()
                att_hits=(att_pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=mask_hits.sum().data.cpu().numpy().item()

                # Log
                total_att_loss+=loss.data.cpu().numpy().item()*len(b)
                total_att_acc+=att_hits.sum().data.cpu().numpy().item()


            else:
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=hits.sum().data.cpu().numpy().item()


            total_num+=len(b)

        if 'all-one' in self.args.similarity_detection:
            choose_att = True
        elif phase=='mcl' and 'no_attention' not in self.args.loss_type:
            if total_att_acc > total_mask_acc:
                choose_att = True

        print('choose_att: ',choose_att)

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

            if phase == 'mcl':
                outputs,masks,outputs_attn=self.model.forward(task,images,s=self.smax,phase=phase,
                                                             similarity=similarity,
                                                             history_mask_pre=history_mask_pre,
                                                             check_federated=check_federated)
                output=outputs[t]

                if outputs_attn is None:
                    loss=self.criterion(output,targets,masks)
                else:
                    output_attn=outputs_attn[t]
                    loss=self.joint_criterion(output,targets,masks,output_attn)

            elif phase == 'transfer' or phase == 'reference':
                outputs=self.model.forward(task,images,s=self.smax,phase=phase,
                                                 pre_mask=pre_mask, pre_task=pre_task)
                output=outputs[t]
                loss=self.transfer_criterion(output,targets)

            if phase=='mcl' and 'no_attention' not in self.args.loss_type and outputs_attn is not None:
                _,att_pred=output_attn.max(1)
                _,mask_pred=output.max(1)

                mask_hits=(mask_pred==targets).float()
                att_hits=(att_pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=mask_hits.sum().data.cpu().numpy().item()

                # Log
                total_att_loss+=loss.data.cpu().numpy().item()*len(b)
                total_att_acc+=att_hits.sum().data.cpu().numpy().item()

            else:
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_mask_loss+=loss.data.cpu().numpy().item()*len(b)
                total_mask_acc+=hits.sum().data.cpu().numpy().item()

            total_num+=len(b)

        if choose_att == True:
            total_loss = total_att_loss
            total_acc = total_att_acc
        else:
            total_loss = total_mask_loss
            total_acc = total_mask_acc

        return total_loss/total_num,total_acc/total_num



    def transfer_criterion(self,outputs,targets,mask=None):
        return self.ce(outputs,targets)


    def joint_criterion(self,outputs,targets,masks,outputs_attn):
        return self.criterion(outputs,targets,masks) + self.args.model_weights*self.ce(outputs_attn,targets)

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


        return self.ce(outputs,targets)+self.lamb*reg

class CheckFederated():
    def __init__(self):
        pass
    def set_similarities(self,similarities):
        self.similarities = similarities

    def fix_length(self):
        return len(self.similarities)

    def get_similarities(self):
        return self.similarities


    def check_t(self,t):
        if t < len([sum(x) for x in zip_longest(*self.similarities, fillvalue=0)]) and [sum(x) for x in zip_longest(*self.similarities, fillvalue=0)][t] > 0:
            return True

        elif np.count_nonzero(self.similarities[t]) > 0:
            return True

        elif t < len(self.similarities[-1]) and self.similarities[-1][t] == 1:
            return True

        return False




########################################################################################################################
