import sys,os,argparse,time
import numpy as np
import torch

import utils

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--experiment',default='',type=str,required=True,choices=['alexmixemnistceleba','mlpmixemnistceleba',
                                                                              'alexceleba','sentiment',
                                                                              'alexmixceleba','mlpmixceleba',
                                                                              'mixemnist','mnist','celeba',
                                                                              'femnist','landmine','cifar10',
                                                                              'cifar100','emnist','mnist2',
                                                                              'pmnist','cifar','mixture'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['adam','random','sgd','sgd-frozen',
                                                                            'lwf','lfl','ewc','imm-mean',
                                                                            'progressive',
                                                                            'alexpathnet','mlppathnet',
                                                                            'imm-mode','sgd-restart',
                                                                            'joint','hat','hat-test'],help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
parser.add_argument('--sbatch',default=0,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--nhid',default=2000,type=int,required=False,help='(default=%(default)d)')
# parser.add_argument('--ntasks',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--idrandom',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--classptask',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--pdrop1',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--pdrop2',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--filters',default=1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--filter_num',default=100,type=int,required=False,help='(default=%(default)d)')
parser.add_argument("--num_class_femnist",default=62,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--dis_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--sim_ntasks',default=10,choices=[10,20],type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--model_weights',default=1,type=float,required=False,help='(default=%(default)d)')
args=parser.parse_args()
if args.output=='':
    args.output='./res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'_'+str(args.note)+'.txt'

performance_output=args.output+'_performance'

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
elif args.experiment=='celeba' or args.experiment=='alexceleba':
    from dataloaders import celeba as dataloader
elif args.experiment=='mlpmixemnistceleba' or args.experiment=='alexmixemnistceleba':
    from dataloaders import mixemnistceleba as dataloader
elif args.experiment=='sentiment':
    from dataloaders import sentiment as dataloader
elif args.experiment=='landmine':
    from dataloaders import landmine as dataloader
elif args.experiment=='cifar10':
    from dataloaders import cifar10 as dataloader
elif args.experiment=='cifar100':
    from dataloaders import cifar100 as dataloader
elif args.experiment=='mnist':
    from dataloaders import mnist as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
if args.approach=='adam':
    from approaches import adam as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
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
elif args.approach=='mlppathnet':
    from approaches import mlppathnet as approach
elif args.approach=='alexpathnet':
    from approaches import alexpathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif args.approach=='joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment=='emnist' \
        or args.experiment=='landmine' or args.experiment=='femnist' or args.experiment=='mlpmixceleba'\
        or args.experiment=='mnist' or args.experiment=='mixemnist' or args.experiment=='mlpmixemnistceleba':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    elif args.approach=='progressive':
        from networks import mlp_progressive as network
    elif args.approach=='mlppathnet':
        from networks import mlp_pathnet as network

    else:
        from networks import mlp as network


elif args.experiment=='sentiment':
    print('sentiment')
    if args.approach=='hat':
        from networks import kimnet_hat as network
    else:
        from networks import kimnet as network

else:
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='alexpathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
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

# Inits
print('Inits...')
if 'sentiment' in args.experiment:
    net=network.Net(inputsize,taskcla,voc_size=voc_size,weights_matrix=weights_matrix,args=args).cuda()
else:
    net=network.Net(inputsize,taskcla,nhid=args.nhid,args=args).cuda()

utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,lr_patience=args.lr_patience,args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
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

        print('task_t', task_t.size())

    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    print('len(ytrain)', len(ytrain))
    print('len(yvalid)', len(yvalid))
    print('len(ytest)', len(data[0]['test']['y']))
    print('xtrain', xtrain.size())

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid,args)
    print('-'*100)
    #
    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        test_loss,test_acc=appr.eval(u,xtest,ytest,args=args)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f',delimiter='\t')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

performance_output_mcl_backward=args.output+'_mcl_backward_performance'
performance_output_mcl_forward=args.output+'_mcl_forward_performance'

with open(performance_output_mcl_backward,'w') as file:
    if args.approach == 'hat' or args.approach == 'sgd' or args.approach == 'joint' \
            or args.approach == 'adam' or args.approach == 'ewc' or args.approach == 'progressive' \
            or 'pathnet' in args.approach:

        for j in range(acc.shape[1]):
            file.writelines(str(acc[-1][j]) + '\n')


with open(performance_output_mcl_forward,'w') as file:
    if args.approach == 'hat' or args.approach == 'sgd' or args.approach == 'joint' \
            or args.approach == 'adam' or args.approach == 'ewc' or args.approach == 'progressive'\
            or 'pathnet' in args.approach:

        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')


with open(performance_output,'w') as file:
    if args.approach == 'hat' or args.approach == 'sgd' or args.approach == 'joint' \
            or args.approach == 'adam' or args.approach == 'ewc' or args.approach == 'progressive'\
            or 'pathnet' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[-1][j]) + '\n')

    elif args.approach == 'sgd-restart':
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')



if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
