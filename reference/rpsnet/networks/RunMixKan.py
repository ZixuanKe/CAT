import sys,os,argparse,time
import numpy as np
import torch

import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy
import statistics
import math
tstart=time.time()

########################################################################################################################


def auto_similarity(task,appr):
    if t > 0:
        for pre_task in range(t+1):
            print('pre_task: ',pre_task)
            print('t: ',t)
            pre_task_torch = torch.autograd.Variable(torch.LongTensor([pre_task]).cuda(),volatile=False)

            gfc1,gfc2 = appr.model.mask(pre_task_torch,s=appr.smax)
            gfc1=gfc1.data.clone()
            gfc2=gfc2.data.clone()

            pre_mask=[gfc1,gfc2]

            if pre_task == t: # the last one

                print('>>> Now Training Phase: {:6s} <<<'.format('reference'))
                appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='reference',args=args,
                                            pre_mask=pre_mask,pre_task=pre_task) # it is actually random mask


            elif pre_task != t:
                print('>>> Now Training Phase: {:6s} <<<'.format('transfer'))
                # appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='transfer',args=args,
                #                             pre_mask=pre_mask,pre_task=pre_task)

            xtest=data[t]['test']['x'].cuda()
            ytest=data[t]['test']['y'].cuda()


            if pre_task == t: # the last one
                test_loss,test_acc=appr.eval(t,xtest,ytest,phase='reference',
                                             pre_mask=pre_mask,pre_task=pre_task)
                test_loss,test_acc = 1,1


            elif pre_task != t:
                test_loss,test_acc=appr.eval(t,xtest,ytest,phase='transfer',
                                             pre_mask=pre_mask,pre_task=pre_task)
                test_loss,test_acc = 0,0
            #
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(t,data[t]['name'],test_loss,100*test_acc))

            acc_transfer[t,pre_task]=test_acc
            lss_transfer[t,pre_task]=test_loss

    print('test_acc: ',acc_transfer[t][:t+1])
    print('test_loss: ',lss_transfer[t][:t+1])

    print('Save at transfer_acc')
    np.savetxt(args.output + '_acc_transfer',acc_transfer,'%.4f',delimiter='\t')

    print('Save at transfer_loss')
    np.savetxt(args.output + '_loss_transfer',lss_transfer,'%.4f',delimiter='\t')


    # normalize the ranking ==============
    similarity = None
    similarity_ranking = None

    if t > 0:
        acc_list = acc_transfer[t][:t] #t from 0
        print('acc_list: ',acc_list)
        # print(type(acc_list))

        if 'auto' in args.note:
            similarity = [0 if (acc_list[acc_id] <= acc_transfer[t][t]) else 1 for acc_id in range(len(acc_list))] # remove all acc < 0.5
        else:
            raise NotImplementedError


        for source_task in range(len(similarity)):
            similarity_transfer[t,source_task]=similarity[source_task]
        print('Save at similarity_transfer')
        np.savetxt(args.output + '_similarity_transfer',similarity_transfer,'%.4f',delimiter='\t')

        Tsim_acc = [acc for acc_id,acc in enumerate(acc_list) if similarity[acc_id]==1]
        print('Tsim_acc: ',Tsim_acc)

        similarity_ranking = [0] * len(Tsim_acc)
        for i, x in enumerate(sorted(range(len(Tsim_acc)), key=lambda y: Tsim_acc[y])):
            similarity_ranking[x] = i

        similarity_ranking = torch.from_numpy(np.array(similarity_ranking)).cuda().long()


    # normalize the ranking ==============
    print('similarity: ',similarity)
    print('similarity_ranking: ',similarity_ranking)

    return similarity,similarity_ranking


def name_similarity(task,norm_transfer_raw,data):
    similarity = None
    if t > 0:
        for pre_task in range(t):
            print('pre_task: ',pre_task)
            print('t: ',t)
            if 'fe-mnist' in data[pre_task]['name'] and 'fe-mnist' in data[t]['name']: #both femnist
                norm_transfer_raw[t,pre_task] = 1

            elif 'celeba' in data[pre_task]['name'] and 'celeba' in data[t]['name']: #both femnist
                norm_transfer_raw[t,pre_task] = 1

            else: # anything else
                norm_transfer_raw[t,pre_task] = 0

        np.savetxt(args.output + '_norm_transfer_raw',norm_transfer_raw,'%.4f',delimiter='\t')

        similarity = norm_transfer_raw[t][:t]
        print('similarity: ',similarity)
    return similarity



def all_one_similarity(task,norm_transfer_raw,data):
    print('all one')
    similarity = None
    if t > 0:
        for pre_task in range(t):
            print('pre_task: ',pre_task)
            print('t: ',t)
            norm_transfer_raw[t,pre_task] = 1

        np.savetxt(args.output + '_norm_transfer_raw',norm_transfer_raw,'%.4f',delimiter='\t')

        similarity = norm_transfer_raw[t][:t]
        print('similarity: ',similarity)
    return similarity

def all_zero_similarity(task,norm_transfer_raw,data):
    print('all zero')

    similarity = None
    if t > 0:
        for pre_task in range(t):
            print('pre_task: ',pre_task)
            print('t: ',t)
            norm_transfer_raw[t,pre_task] = 0

        np.savetxt(args.output + '_norm_transfer_raw',norm_transfer_raw,'%.4f',delimiter='\t')

        similarity = norm_transfer_raw[t][:t]
        print('similarity: ',similarity)
    return similarity

########################################################################################################################



# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--experiment',default='',type=str,required=True,choices=['alexmixemnistceleba','mlpmixemnistceleba','mixemnist','sentiment','alexmixceleba','mlpmixceleba','mnist','celeba','femnist','landmine','cifar10','cifar100','emnist','mnist2','pmnist','cifar','mixture'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['MixKan','kan','random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                            'imm-mode','sgd-restart',
                                                                            'joint','hat','hat-test'],help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
parser.add_argument('--nhid',default=2000,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--ntasks',default=10,choices=[5,10],type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--idrandom',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--classptask',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--pdrop1',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--pdrop2',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--filter_num',default=100,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--A',default=2,type=float,required=False,help='(default=%(default)d)')
parser.add_argument('--lamb',default=-1,type=float,required=False,help='(default=%(default)d)')
parser.add_argument("--alpha", default=0.7,type=float,required=False,help='(default=%(default)d)')
parser.add_argument("--T",default=20,type=float,required=False,help='(default=%(default)d)')
parser.add_argument("--loss_k",default=1,type=float,required=False,help='(default=%(default)d)')
parser.add_argument("--num_class_femnist",default=62,type=int,required=False,help='(default=%(default)d)')
parser.add_argument("--smax",default=400,type=int,required=False,help='(default=%(default)d)')
parser.add_argument("--n_head",default=1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr_kt',default=0.025,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--nepochs_kt',default=300,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_patience_kt',default=10,type=int,required=False,help='(default=%(default)f)')


args=parser.parse_args()
if args.output=='':
    args.output='./res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+str(args.note)+'.txt'


print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='emnist':
    from dataloaders import emnist as dataloader
elif args.experiment=='femnist':
    from dataloaders import femnist as dataloader
elif args.experiment=='mixemnist':
    from dataloaders import mixemnist as dataloader
elif args.experiment=='alexmixceleba' or args.experiment=='mlpmixceleba':
    from dataloaders import mixceleba as dataloader
elif args.experiment=='cifar10':
    from dataloaders import cifar10 as dataloader
elif args.experiment=='cifar100':
    from dataloaders import cifar100 as dataloader
elif args.experiment=='mlpmixemnistceleba' or args.experiment=='alexmixemnistceleba':
    from dataloaders import mixemnistceleba as dataloader
elif args.experiment=='sentiment':
    from dataloaders import sentiment as dataloader
elif args.experiment=='mnist':
    from dataloaders import mnist as dataloader
elif args.experiment=='celeba':
    from dataloaders import celeba as dataloader
elif args.experiment=='landmine':
    from dataloaders import landmine as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach

if args.approach=='kan':
    from approaches import kan as approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='MixKan':
    from approaches import MixKan as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif args.approach=='joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment=='emnist' \
        or args.experiment=='landmine' or args.experiment=='femnist'or args.experiment=='celeba'\
        or args.experiment=='mnist' or args.experiment=='mlpmixceleba'or args.experiment=='mixemnist' \
        or args.experiment=='mlpmixemnistceleba':
    print(args.experiment)
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    elif args.approach=='kan'\
            or 'by-name' in args.note or 'all-zero' in args.note  or 'all-one' in args.note:
        from networks import MlpKan as network

    elif 'auto' in args.note:
        from networks import MlpKan as network

    else:
        from networks import mlp as network

elif args.experiment=='sentiment':
    if args.approach=='kan':
        from networks import KimnetKan as network

else:

    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    elif args.approach=='kan'\
            or 'by-name' in args.note  or 'all-zero' in args.note  \
            or 'all-one' in args.note:
        from networks import AlexnetKan as network
    elif 'auto' in args.note:
        from networks import AlexnetMixTaskKan as network
    else:
        from networks import alexnet as network

########################################################################################################################

# Load
print('Load data...')
if 'sentiment' in args.experiment:
    data,taskcla,inputsize,voc_size,weights_matrix=dataloader.get(seed=args.seed,args=args)
else:
    data,taskcla,inputsize=dataloader.get(seed=args.seed,args=args)
print('Input size =',inputsize,'\nTask info =',taskcla)


print('taskcla: ',len(taskcla))
if 'nofemnist' in args.note:
    c = 0
    taskcla_new = []
    data_new = []
    for i,(t,ncla) in enumerate(taskcla):
        if 'fe-mnist' in data[t]['name']:
            continue
        else:
            taskcla_new.append((c,taskcla[i][1]))
            data_new.append(data[t])
            c+=1
    data = data_new
    taskcla = taskcla_new
print('taskcla: ',len(taskcla))
print('taskcla: ',taskcla)


# Inits
print('Inits...')
if 'sentiment' in args.experiment:
    net=network.Net(inputsize,taskcla,voc_size=voc_size,weights_matrix=weights_matrix,nhid=args.nhid).cuda()
else:
    net=network.Net(inputsize,taskcla,nhid=args.nhid,args=args).cuda()
utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,lr_patience=args.lr_patience,
                   nepochs_kt=args.nepochs_kt,lr_kt=args.lr_kt,lr_patience_kt=args.lr_patience_kt,
                   args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc_ac=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss_ac=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

acc_mcl=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss_mcl=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

acc_an=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss_an=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

unit_overlap_sum_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
norm_transfer_raw=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
norm_transfer_one=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
norm_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
acc_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
acc_reference=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
similarity_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)


similarity = None
history_mask_back = []
history_mask_pre = []

for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    if t==0:
        candidate_phases = ['mcl']
    elif t>0:
        if 'pipeline' in args.note:
            candidate_phases = ['kt','mcl']
        else:
            candidate_phases = ['mcl']

    else:
        raise NotImplementedError


    for candidate_phase in candidate_phases:

        if 'pipeline' in args.note:

            if candidate_phase == 'kt' and 'auto' in args.note:
                similarity,similarity_ranking = auto_similarity(task,appr)
                net.ranking = similarity_ranking

            elif candidate_phase == 'kt' and 'by-name' in args.note:
                similarity = name_similarity(task,norm_transfer_raw,data)
            elif candidate_phase == 'kt' and 'all-one' in args.note:
                similarity = all_one_similarity(task,norm_transfer_raw,data)
            elif candidate_phase == 'kt' and 'all-zero' in args.note:
                similarity = all_zero_similarity(task,norm_transfer_raw,data)

        else:
            if candidate_phase == 'mcl' and 'auto' in args.note:
                similarity,similarity_ranking = auto_similarity(task,appr)
                net.ranking = similarity_ranking
            elif candidate_phase == 'mcl' and 'by-name' in args.note:
                similarity = name_similarity(task,norm_transfer_raw,data)
            elif candidate_phase == 'mcl' and 'all-one' in args.note:
                similarity = all_one_similarity(task,norm_transfer_raw,data)
            elif candidate_phase == 'mcl' and 'all-zero' in args.note:
                similarity = all_zero_similarity(task,norm_transfer_raw,data)

        # else:
        #     raise NotImplementedError
        print('>>> Now Training Phase: {:6s} <<<'.format(candidate_phase))


        appr.train(task,xtrain,ytrain,xvalid,yvalid,candidate_phase,args,
                   similarity=similarity,history_mask_back=history_mask_back,
                   history_mask_pre=history_mask_pre)

        print('-'*100)
        #
        # Test
        if candidate_phase == 'kt':
            for u in range(t+1):
                xtest=data[u]['test']['x'].cuda()
                ytest=data[u]['test']['y'].cuda()
                test_loss,test_acc=appr.eval(u,xtest,ytest,candidate_phase,args,similarity=similarity,history_mask_pre=history_mask_pre)

                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
                acc_ac[t,u]=test_acc
                lss_ac[t,u]=test_loss

            # Save
            print('Save at '+args.output + '_' + candidate_phase)
            np.savetxt(args.output + '_' + candidate_phase,acc_ac,'%.4f',delimiter='\t')

        elif candidate_phase == 'mcl':
            history_mask_back.append(dict((k, v.data.clone()) for k, v in appr.mask_back.items()) )
            history_mask_pre.append([m.data.clone() for m in appr.mask_pre])

            for u in range(t+1):
                xtest=data[u]['test']['x'].cuda()
                ytest=data[u]['test']['y'].cuda()


                test_loss,test_acc=appr.eval(u,xtest,ytest,candidate_phase,args,similarity=similarity,history_mask_pre=history_mask_pre)
                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))

                acc_mcl[t,u]=test_acc
                lss_mcl[t,u]=test_loss

            # Save
            print('Save at '+args.output + '_' + candidate_phase)
            np.savetxt(args.output + '_' + candidate_phase,acc_mcl,'%.4f',delimiter='\t')


# Done
print('*'*100)
print('Accuracies =')
for i in range(acc_mcl.shape[0]):
    print('\t',end='')
    for j in range(acc_mcl.shape[1]):
        print('{:5.1f}% '.format(100*acc_mcl[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


performance_output_mcl_backward=args.output+'_mcl_backward_performance'
performance_output_mcl_forward=args.output+'_mcl_forward_performance'



with open(performance_output_mcl_backward,'w') as file:
    if args.approach == 'kan' or args.approach == 'MixKan':
        for j in range(acc_mcl.shape[1]):
            file.writelines(str(acc_mcl[-1][j]) + '\n')
    else:
        raise NotImplementedError

with open(performance_output_mcl_forward,'w') as file:
    if args.approach == 'kan' or args.approach == 'MixKan':
        for j in range(acc_mcl.shape[1]):
            file.writelines(str(acc_mcl[j][j]) + '\n')
    else:
        raise NotImplementedError


if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc_mcl[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss_mcl[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)






# scipy eigs =======

        # eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(tournament, k=1, which='LM')
        # p_eigenvectors = eigenvectors[:,torch.argmax(eigenvalues,dim=0)[0]]
        # similarity = [x/sum(p_eigenvectors) for x in p_eigenvectors]