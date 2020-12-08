import sys,os,argparse,time
import numpy as np
import torch

import utils
from config import set_args

tstart=time.time()
args = set_args()

# Arguments
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
if args.experiment=='mixemnist':
    from dataloaders import mixemnist as dataloader
elif args.experiment=='mixceleba':
    from dataloaders import mixceleba as dataloader


# Args -- Approach
if 'random_ncl' in args.approach:
    from approaches import random_ncl as approach
elif args.approach=='mlp_ncl' or args.approach=='cnn_ncl':
    from approaches import ncl as approach
elif 'one' in args.approach:
    from approaches import one as approach
elif 'frozen_ncl' in args.approach:
    from approaches import frozen_ncl as approach
elif 'lwf_ncl' in args.approach:
    from approaches import lwf_ncl as approach
elif 'lfl_ncl' in args.approach:
    from approaches import lfl_ncl as approach
elif 'ewc_ncl' in args.approach:
    from approaches import ewc_ncl as approach
elif 'imm_mean_ncl' in args.approach:
    from approaches import imm_mean_ncl as approach
elif 'imm_mode_ncl' in args.approach:
    from approaches import imm_mode_ncl as approach
elif 'progressive_ncl' in args.approach:
    from approaches import progressive_ncl as approach
elif args.approach=='mlp_pathnet_ncl':
    from approaches import mlp_pathnet_ncl as approach
elif args.approach=='cnn_pathnet_ncl':
    from approaches import cnn_pathnet_ncl as approach
elif 'hat_ncl' in args.approach:
    from approaches import hat_ncl as approach
elif 'mtl' in args.approach:
    from approaches import mtl as approach

# Args -- Network
if 'mlp_hat' in args.approach:
    from networks import mlp_hat as network
elif 'mlp_progressive' in args.approach:
    from networks import mlp_progressive as network
elif 'mlp_pathnet' in args.approach:
    from networks import mlp_pathnet as network
elif 'cnn_lfl' in args.approach:
    from networks import cnn_lfl as network
elif 'cnn_hat' in args.approach:
    from networks import cnn_hat as network
elif 'cnn_progressive' in args.approach:
    from networks import cnn_progressive as network
elif 'cnn_pathnet' in args.approach:
    from networks import cnn_pathnet as network

elif 'cnn' in args.approach:
    from networks import cnn as network
elif 'mlp' in args.approach:
    from networks import mlp as network
########################################################################################################################

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed,args=args)
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla,nhid=args.nhid,args=args).cuda()

# utils.print_model_report(net)

appr=approach.Appr(net,args=args)
# print(appr.criterion)
# utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if 'mtl' in args.approach:
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

    # print('len(ytrain)', len(ytrain))
    # print('len(yvalid)', len(yvalid))
    # print('len(ytest)', len(data[0]['test']['y']))
    # print('xtrain', xtrain.size())

    # Train

    appr.train(task,xtrain,ytrain,xvalid,yvalid,args)
    print('-'*100)
    #
    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()

        xvalid=data[u]['valid']['x'].cuda()
        yvalid=data[u]['valid']['y'].cuda()

        # test_loss,test_acc=appr.eval(u,xtest,ytest,args=args)
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

with open(performance_output_mcl_backward,'w') as file_backward, open(performance_output_mcl_forward,'w') as file_forward:
    for j in range(acc.shape[1]):
        file_backward.writelines(str(acc[-1][j]) + '\n')

    for j in range(acc.shape[1]):
        file_forward.writelines(str(acc[j][j]) + '\n')


########################################################################################################################
