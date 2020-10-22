import os
# import shutil

import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Learner():
    def __init__(self,task,model,args,trainloader,testloader,old_model,use_cuda, path, fixed_path, train_path, infer_path):
        self.model=model
        self.args=args
        self.title='cifar-100-' + self.args.arch
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.best_acc = 0 
        self.testloader=testloader
        self.start_epoch=self.args.start_epoch
        self.test_loss=0.0
        self.path = path
        self.fixed_path = fixed_path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0
        self.old_model=old_model
        if self.args.sess > 0: self.old_model.eval()


        trainable_params = []
        
        # if(self.args.dataset=="MNIST"):
        params_set = [self.model.mlp1, self.model.mlp2]
        # else:
        #     params_set = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4, self.model.conv5, self.model.conv6, self.model.conv7, self.model.conv8, self.model.conv9]
        for j, params in enumerate(params_set): 
            for i, param in enumerate(params):
                if(i==self.args.M):
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                else:
                    if(self.train_path[j,i]==1):
                        p = {'params': param.parameters()}
                        trainable_params.append(p)
                    else:
                        param.requires_grad = False
                    
                    
        p = {'params': self.model.final_layers[task].parameters()}
        trainable_params.append(p)
        print("Number of layers being trained : " , len(trainable_params))
        
        
#         self.optimizer = optim.Adadelta(trainable_params)
#         self.optimizer = optim.SGD(trainable_params, lr=self.args.lr, momentum=0.96, weight_decay=0)
        self.optimizer = optim.Adam(trainable_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        



    def learn(self, task, taskcla):
        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args.resume), 'Error: no checkpoint directory found!'
            self.args.checkpoint = os.path.dirname(self.args.resume)
            checkpoint = torch.load(self.args.resume)
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(self.args.checkpoint, 'log.txt'), title=self.title, resume=True)
        else:
            logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_'+str(self.args.test_case)+'_log.txt'), title=self.title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])
        # if self.args.evaluate:
        #     print('\nEvaluation only')
        #     self.test(self.start_epoch)
        #     print(' Test Loss:  %.8f, Test Acc:  %.2f' % (self.test_loss, self.test_acc))
        #     return


        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(epoch)

            # print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))
            self.train(epoch, self.infer_path, task, taskcla)
            self.test(epoch, self.infer_path, task)
            

            # append logger file
            logger.append([self.state['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc, self.best_acc])

            # save model
            is_best = self.test_acc > self.best_acc
            self.best_acc = max(self.test_acc, self.best_acc)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'acc': self.test_acc,
                    'best_acc': self.best_acc,
                    'optimizer' : self.optimizer.state_dict(),
            }, is_best, checkpoint=self.args.savepoint,filename='session_'+str(self.args.sess)+'_' + str(self.args.test_case)+"_"+self.args.experiment+"_"+str(self.args.dis_ntasks+self.args.sim_ntasks)+"_"+str(self.args.idrandom)+'_checkpoint.pth.tar',session=self.args.sess, test_case=self.args.test_case)
        
        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)

    def train(self, epoch, path, last, taskcla):
        # switch to train mode

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            
            # measure data loading time
            data_time.update(time.time() - end)

            current_task_id = last
            class_per_task = [x[1] for x in taskcla]
            # print (class_per_task)
            num_class = sum(class_per_task) # total number
            offset = sum(class_per_task[:current_task_id])

            inputs = inputs.to(device)
            targets = targets.to(device)


            current_targets_one_hot = torch.FloatTensor(inputs.shape[0], class_per_task[current_task_id]).to(device)
            current_targets_one_hot.zero_()
            current_targets_one_hot.scatter_(1, targets[:,None], 1)

            previous_targets_one_hot = torch.FloatTensor(inputs.shape[0], offset).to(device)
            previous_targets_one_hot.zero_()

            future_targets_one_hot = torch.FloatTensor(inputs.shape[0], sum(class_per_task[current_task_id+1:])).to(device)
            future_targets_one_hot.zero_()

            targets_one_hot = torch.cat([previous_targets_one_hot,current_targets_one_hot,future_targets_one_hot],1)
            targets_one_hot = targets_one_hot.to(device)
            # print (targets_one_hot.size())
            # if self.use_cuda:
            #     inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(), targets.cuda()
            inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot), torch.autograd.Variable(targets)
            
            # compute output
            outputs = self.model(inputs, path, last) # -1
            
            preds = torch.cat( (torch.zeros(inputs.shape[0],sum(class_per_task[:current_task_id]),device=device),outputs,torch.zeros(inputs.shape[0],sum(class_per_task[current_task_id+1:]),device=device)), dim=1)
            
            # print (preds.size(), outputs.size())
            # preds=outputs.masked_select(targets_one_hot.eq(1))
            # print ('targets:', last, targets.size(), outputs.size())
            
            # targets_one_hot = torch.FloatTensor(inputs.shape[0], outputs.size(-1)).to(device)
            # targets_one_hot.zero_()
            # targets_one_hot.scatter_(1, targets[:,None], 1)

            # if self.use_cuda:
            #     targets_one_hot, targets = torch.autograd.Variable(targets_one_hot), torch.autograd.Variable(targets)
            
            # print (targets)
            tar_ce=targets
            pre_ce=outputs.clone()

            # pre_ce=pre_ce[:,0:self.args.class_per_task*(1+self.args.sess)] ## what
            pre_ce=pre_ce[:,0:sum(class_per_task[:current_task_id+1])] #including current_task_id
            
            # print ('ce:', pre_ce, tar_ce)
            loss = F.cross_entropy(pre_ce,tar_ce)
            loss_dist = 0

            # distillation loss
            # if self.args.sess > 0:
            #     outputs_old=self.old_model(inputs, path, last)
            #     outputs_old = torch.cat( (torch.zeros(inputs.shape[0],sum(class_per_task[:current_task_id]),device=device),outputs_old,torch.zeros(inputs.shape[0],sum(class_per_task[current_task_id+1:]),device=device)), dim=1)
                
            #     t_one_hot=targets_one_hot.clone()
            #     preds_one_hot = preds.clone()
            #     # t_one_hot[:,0:self.args.class_per_task*self.args.sess]=outputs_old[:,0:self.args.class_per_task*self.args.sess]
                
            #     t_one_hot[:,0:sum(class_per_task[:current_task_id+1])]=outputs_old[:,0:sum(class_per_task[:current_task_id+1])]

            #     if(self.args.sess in range(1+self.args.jump)):
            #         cx = 1
            #     else:
            #         cx = self.args.rigidness_coff*(self.args.sess-self.args.jump)
            #     # print (pre_ce.size(), preds_one_hot.size(), outputs.size(), t_one_hot.size())
            #     loss_dist = ( cx/self.args.train_batch*1.0)* torch.sum(F.kl_div(F.log_softmax(preds/2.0,dim=1),F.softmax(t_one_hot/2.0,dim=1),reduce=False).clamp(min=0.0))

            loss+=loss_dist




            # measure accuracy and record loss
            # if(self.args.dataset=="MNIST"):
            #     prec1, prec5 = accuracy(output=outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], target=targets.cuda().data, topk=(1, 1))
            # else:
            #     prec1, prec5 = accuracy(output=outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], target=targets.cuda().data, topk=(1, 5))
            
            # if(self.args.dataset=="MNIST"):
            if self.use_cuda:
                prec1, prec5 = accuracy(output=outputs.data, target=targets.data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output=outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], target=targets.data, topk=(1, 1))
            # else:
            #     if self.use_cuda:
            #         prec1, prec5 = accuracy(output=outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], target=targets.cuda().data, topk=(1, 5))
            #     else:
            #         prec1, prec5 = accuracy(output=outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], target=targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
#             bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | Dist: {loss_dist:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
#                         batch=batch_idx + 1,
#                         size=len(self.trainloader),
# #                         data=data_time.avg,
# #                         bt=batch_time.avg,
#                         total=bar.elapsed_td,
# #                         eta=bar.eta_td,
#                         loss=losses.avg,
#                         loss_dist=loss_dist,
#                         top1=top1.avg,
#                         top5=top5.avg
#                         )
#             bar.next()
        # bar.finish()
        self.train_loss,self.train_acc=losses.avg, top1.avg

   
    
    def test(self, epoch, path, last):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time.time() - end)


            # targets_one_hot = torch.FloatTensor(inputs.shape[0], self.args.num_class)
            # targets_one_hot.zero_()
            # targets_one_hot.scatter_(1, targets[:,None], 1)
            # if self.use_cuda:
            #     inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
            # inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            

            

            outputs = self.model(inputs, path, last)

            loss = F.cross_entropy(outputs, targets)



            # measure accuracy and record loss
            # if(self.args.dataset=="MNIST"):
            if self.use_cuda:
                prec1, prec5 = accuracy(outputs.cuda().data, targets.cuda().data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], targets.data, topk=(1, 1))
            # else:    
            #     prec1, prec5 = accuracy(outputs.data[:,0:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
                        total=bar.elapsed_td,
#                         eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg

    def save_checkpoint(self,state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',session=0, test_case=0):
#         filepath = os.path.join(checkpoint, filename)
#         torch.save(state, filepath)
        if is_best:
            torch.save(state, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+"_"+self.args.experiment+"_"+str(self.args.dis_ntasks+self.args.sim_ntasks)+"_"+str(self.args.idrandom)+'_model_best.pth.tar'))
#             shutil.copyfile(filepath, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+'_model_best.pth.tar') )

    def adjust_learning_rate(self, epoch):
        schedule = [6, 8, 16]
        if epoch in schedule:#self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']



    def get_confusion_matrix(self, path):
        
        confusion_matrix = torch.zeros(100, 100)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                inputs = inputs.to(device)#.cuda()
                targets = targets.to(device)#.cuda()
                outputs = self.model(inputs, path, -1)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

