import sys,os,argparse,time
import numpy as np
import torch

import util
from util import get_path, get_best_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy
import statistics
import math
import scipy.stats as ss


from rps_net import RPS_net_mlp, RPS_net_cifar
from learner import Learner
import copy
import pickle
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F

tstart=time.time()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# conda activate pytorch

########################################################################################################################
'''
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random0,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --sim_ntasks=10 --classptask=5 --idrandom 0 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random1,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --sim_ntasks=10 --classptask=5 --idrandom 1 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random2,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --sim_ntasks=10 --classptask=5 --idrandom 2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random3,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --sim_ntasks=10 --classptask=5 --idrandom 3 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random4,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --sim_ntasks=10 --classptask=5 --idrandom 4 --lr 0.001 --lr_patience 10 --n_head 5

CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random0,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5 --dis_ntasks=20 --classptask=2 --idrandom 0 --pdrop1 0.2 --pdrop2 0.2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random1,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5 --dis_ntasks=20 --classptask=2 --idrandom 1 --pdrop1 0.2 --pdrop2 0.2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random2,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5 --dis_ntasks=20 --classptask=2 --idrandom 2 --pdrop1 0.2 --pdrop2 0.2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random3,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5 --dis_ntasks=20 --classptask=2 --idrandom 3 --pdrop1 0.2 --pdrop2 0.2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mixemnist --approach=MixKan --note random4,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5 --dis_ntasks=20 --classptask=2 --idrandom 4 --pdrop1 0.2 --pdrop2 0.2 --lr 0.001 --lr_patience 10 --n_head 5

CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random0,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --classptask=10 --idrandom 0 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random1,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --classptask=10 --idrandom 1 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=1 python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random2,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --classptask=10 --idrandom 2 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=0 python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random3,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --classptask=10 --idrandom 3 --lr 0.001 --lr_patience 10 --n_head 5
CUDA_VISIBLE_DEVICES=0 python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random4,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5 --dis_ntasks=10 --classptask=10 --idrandom 4 --lr 0.001 --lr_patience 10 --n_head 5

python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random0,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5,ntask30 --dis_ntasks=20 --classptask=5 --idrandom 0 --lr 0.001 --lr_patience 10 --n_head 5
python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random1,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5,ntask30 --dis_ntasks=20 --classptask=5 --idrandom 1 --lr 0.001 --lr_patience 10 --n_head 5
python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random2,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5,ntask30 --dis_ntasks=20 --classptask=5 --idrandom 2 --lr 0.001 --lr_patience 10 --n_head 5
python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random3,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5,ntask30 --dis_ntasks=20 --classptask=5 --idrandom 3 --lr 0.001 --lr_patience 10 --n_head 5
python run_rps_in_our_framework.py --experiment=mlpmixceleba --approach=MixKan --note random4,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5,ntask30 --dis_ntasks=20 --classptask=5 --idrandom 4 --lr 0.001 --lr_patience 10 --n_head 5

'''

'''


'''
########################################################################################################################


def auto_similarity(task,appr):
    if t > 0:


        for pre_task in range(t+1):
            print('pre_task: ',pre_task)
            print('t: ',t)
            pre_task_torch = torch.autograd.Variable(torch.LongTensor([pre_task]).cuda(),volatile=False)

            gfc1,gfc2 = appr.model.mask(pre_task_torch,s=appr.smax)
            gfc1=gfc1.detach()
            gfc2=gfc2.detach()
            pre_mask=[gfc1,gfc2]

            if pre_task == t: # the last one

                print('>>> Now Training Phase: {:6s} <<<'.format('reference'))
                appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='reference',args=args,
                                            pre_mask=pre_mask,pre_task=pre_task) # it is actually random mask
            elif pre_task != t:
                print('>>> Now Training Phase: {:6s} <<<'.format('transfer'))
                appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='transfer',args=args,
                                            pre_mask=pre_mask,pre_task=pre_task)

            xtest=data[t]['test']['x'].cuda()
            ytest=data[t]['test']['y'].cuda()


            if pre_task == t: # the last one
                test_loss,test_acc=appr.eval(t,xtest,ytest,phase='reference',
                                             pre_mask=pre_mask,pre_task=pre_task)
            elif pre_task != t:
                test_loss,test_acc=appr.eval(t,xtest,ytest,phase='transfer',
                                             pre_mask=pre_mask,pre_task=pre_task)


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
    similarity = [0]
    ranking = None
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


        acc_to_rank = []
        for s in range(len(similarity)):
            acc_to_rank.append(acc_transfer[t][s])

        ranking = torch.from_numpy(
            np.append(ss.rankdata(acc_to_rank,method='min'),0)).long()



    # normalize the ranking ==============
    print('similarity: ',similarity)
    return similarity,ranking

def read_baseline_similarity():

    similarity = [0]
    ranking = None
    if t > 0:
        if 'ntask30' in args.note:
            f_name = './baseline-sim/'+str(args.experiment)+'_MixKan_0_random'+str(args.idrandom)+',small,auto,baseline,ntask30,lr0.025,patient10,nhead5.txt_similarity_transfer'
        else:
            f_name = './baseline-sim/'+str(args.experiment)+'_MixKan_0_random'+str(args.idrandom)+',small,auto,baseline,lr0.025,patient10,nhead5.txt_similarity_transfer'

        with open(f_name,'r') as f:
            similarity = [int(float(_))for _ in f.readlines()[t].split('\t')[:t]]
            ranking=similarity

    print('similarity: ',similarity)
    return similarity,ranking




def name_similarity(task,norm_transfer_raw,data):
    similarity = [0]
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
    similarity = [0]
    if t > 0:
        for pre_task in range(t):
            print('pre_task: ',pre_task)
            print('t: ',t)
            norm_transfer_raw[t,pre_task] = 1

        np.savetxt(args.output + '_norm_transfer_raw',norm_transfer_raw,'%.4f',delimiter='\t')

        similarity = norm_transfer_raw[t][:t]
        print('similarity: ',similarity)
    return similarity,similarity

def all_zero_similarity(task,norm_transfer_raw,data):
    print('all zero')

    similarity = [0]
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
parser.add_argument('--lr',default=0.001,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
parser.add_argument('--nhid',default=2000,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--dis_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--sim_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--model_weights',default=1,type=float,required=False,help='(default=%(default)d)')

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

parser.add_argument('--M',default=8,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--L',default=9,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--N',default=4,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--start_epoch',default=0,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--sess',default=0,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--num_class',default=10,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--class_per_task',default=5,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--gamma',default=0.5,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--rigidness_coff',default=2.5,type=float,required=False,help='(default=%(default)f)')

parser.add_argument('--arch',default='rps',type=str,required=False)
parser.add_argument('--resume',default=False,type=bool,required=False)
parser.add_argument('--epochs',default=200,type=int,required=False,help='(default=%(default)d)')



args=parser.parse_args()

args.checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
args.labels_data = "prepare/cifar100_10.pkl"
args.savepoint = ""
args.jump = 1
args.train_batch = 64
args.test_batch = 64

if args.output=='':
    # args.output='./res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+str(args.note)+'.txt'
    args.output='./res/'+args.experiment+'_'+'rps'+'_'+str(args.seed)+'_random'+str(args.idrandom)+'_ntasks'+str(args.dis_ntasks+args.sim_ntasks)+'.txt'


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
# util.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,lr_patience=args.lr_patience,
                   nepochs_kt=args.nepochs_kt,lr_kt=args.lr_kt,lr_patience_kt=args.lr_patience_kt,
                   args=args)
print(appr.criterion)
# util.print_optimizer_config(appr.optimizer)
print('-'*100)

check_federated = approach.CheckFederated()



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


similarity = [0]
history_mask_back = []
history_mask_pre = []
similarities = []
rankings = []


timers_train = []
timers_test = []

start_sess = 0

model = RPS_net_mlp(args,taskcla).to(device)#.cuda() 
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

if not os.path.isdir("models/CIFAR100/"+args.checkpoint.split("/")[-1]):
    mkdir_p("models/CIFAR100/"+args.checkpoint.split("/")[-1])
args.savepoint = "models/CIFAR100/"+args.checkpoint.split("/")[-1]

for t,ncla in taskcla:
    for test_case in range(1):
        print('*'*100)
        print('Task {:2d} ({:s})'.format(t,data[t]['name']))
        print('*'*100)

        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t
        ses=t # session is task

        # print ('input:', xtrain.shape, ytrain.shape)
        
        # start_sess = int(sys.argv[2])
        # test_case = sys.argv[1]
        # print (start_sess, test_case)
        # test_case = task
        args.test_case = test_case

        inds_all_sessions=pickle.load(open(args.labels_data,'rb'))
        
    
        if(ses==0):
            path = get_path(args.L,args.M,args.N)*0 
            path[:,0] = 1
            fixed_path = get_path(args.L,args.M,args.N)*0 
            train_path = path.copy()
            infer_path = path.copy()
        else:
            load_test_case = get_best_model(ses-1, args.checkpoint)
            if(ses%args.jump==0):   #get a new path
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(ses-1)+"_"+str(load_test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy")
                path = get_path(args.L,args.M,args.N)
                train_path = get_path(args.L,args.M,args.N)*0 
            else:
                if((ses//args.jump)==0):
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_test_case_x = get_best_model((ses//args.jump)*args.jump-1, args.checkpoint)
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str((ses//args.jump)*args.jump-1)+"_"+str(load_test_case_x)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy")
                path = np.load(args.checkpoint+"/path_"+str(ses-1)+"_"+str(load_test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy")
                train_path = get_path(args.L,args.M,args.N)*0 
            infer_path = get_path(args.L,args.M,args.N)*0 
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        train_path[j,i]=1
                    if(fixed_path[j,i]==1 or path[j,i]==1):
                        infer_path[j,i]=1
            
        np.save(args.checkpoint+"/path_"+str(ses)+"_"+str(test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy", path)
        
        
        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n",path)
        print("fixed_path\n",fixed_path)
        print("train_path\n", train_path)
        
    
        # ind_this_session=inds_all_sessions[ses]    
        # ind_trn= ind_this_session['curent']
        # if ses > 0: ind_trn = np.concatenate([ind_trn,  np.tile(inds_all_sessions[ses-1]['exmp'],int(1))]).ravel()
        # ind_tst=inds_all_sessions[ses]['test']

        
        # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train,ind=ind_trn)
        # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,num_workers=args.workers)
        # testset = dataloader(root='./data', train=False, download=False, transform=transform_test,ind=ind_tst)
        # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
        trainset = torch.utils.data.TensorDataset(xtrain, ytrain) # create your datset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True) # create your data 

        testset = torch.utils.data.TensorDataset(xvalid, yvalid) # create your datset
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch)
        
        args.sess=ses
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1)+'_'+str(load_test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+'_model_best.pth.tar')
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best['state_dict'])


        main_learner=Learner(task, model=model,args=args,trainloader=trainloader,
                             testloader=testloader,old_model=copy.deepcopy(model),
                             use_cuda=use_cuda, path=path, 
                             fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)
        main_learner.learn(task, taskcla)

        
        if(ses==0):
            fixed_path = path.copy()
        else:
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        fixed_path[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy", fixed_path)
        
        
        best_model = get_best_model(ses, args.checkpoint)
    
        cfmat = main_learner.get_confusion_matrix(infer_path)
        np.save(args.checkpoint+"/confusion_matrix_"+str(ses)+"_"+str(test_case)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy", cfmat)

        # path = util.get_path(args.L,args.M,args.N)*0 
        # path[:,0] = 1
        # fixed_path = util.get_path(args.L,args.M,args.N)*0 
        # train_path = path.copy()
        # infer_path = path.copy()

        # model = RPS_net_mlp(args,taskcla).to(device)#.cuda() 


        # train_dataset = torch.utils.data.TensorDataset(xtrain, ytrain) # create your datset
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True) # create your data


        # test_dataset = torch.utils.data.TensorDataset(xvalid, yvalid) # create your datset
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
        
        
        # main_learner=Learner(model=model,args=args,trainloader=train_loader,
        #                          testloader=test_loader,old_model=copy.deepcopy(model),
        #                          use_cuda=use_cuda, path=path, 
        #                          fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)

        
        # main_learner.learn(task)

        
        # model = main_learner.model

        # appr.train(task,xtrain,ytrain,xvalid,yvalid,candidate_phase,args,
        #            similarity=similarity,history_mask_back=history_mask_back,
        #            history_mask_pre=history_mask_pre,check_federated=check_federated)

        print('-'*100)
            
        t2  = time.time()
            
        
        t3 = time.time()
    #    history_mask_back.append(dict((k, v.data.clone()) for k, v in appr.mask_back.items()) )
    #    history_mask_pre.append([m.data.clone() for m in appr.mask_pre])

        for u in range(t+1):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            print ('test:', xtest.size(), ytest.size())
            # main_learner.test(200,infer_path,u)

            
            path_model=os.path.join(args.savepoint, 'session_'+str(ses)+'_'+str(best_model)+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+'_model_best.pth.tar')
            prev_best=torch.load(path_model)
            # model_infer = copy.deepcopy(model)
            model.load_state_dict(prev_best['state_dict'])
            
            if ses > 0:
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(u)+"_"+str(get_best_model(u, args.checkpoint))+"_"+args.experiment+"_"+str(args.dis_ntasks+args.sim_ntasks)+"_"+str(args.idrandom)+".npy")
                path = get_path(args.L,args.M,args.N)
                train_path = get_path(args.L,args.M,args.N)*0 
                infer_path = get_path(args.L,args.M,args.N)*0 
                for j in range(args.L):
                    for i in range(args.M):
                        if(fixed_path[j,i]==0 and path[j,i]==1):
                            train_path[j,i]=1
                        if(fixed_path[j,i]==1 or path[j,i]==1):
                            infer_path[j,i]=1

            outputs = model(xtest, infer_path, u)
            
            loss = F.cross_entropy(outputs, ytest)

            prec1, prec5 = accuracy(outputs.data, ytest.data, topk=(1, 1))

            # test_loss,test_acc=appr.eval(u,xtest,ytest,candidate_phase,args,similarity=similarity,history_mask_pre=history_mask_pre,check_federated=check_federated)
            # print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            test_acc = prec1
            test_loss = loss
            
            acc_mcl[t,u]=test_acc
            lss_mcl[t,u]=test_loss

        # Save
        candidate_phase = 'rps'
        print('Save at '+args.output + '_' + candidate_phase)
        np.savetxt(args.output + '_' + candidate_phase,acc_mcl,'%.4f',delimiter='\t')

        t4 = time.time()
        # timers_test.append((t4-t3)*t)
        # timers_train.append( (t2-t1)*200 )


# Done
#print ('timer train:', np.mean(timers_train)/3600)
#print ('timer test:', timers_test[-1])

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
