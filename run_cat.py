import sys,os,argparse,time
import numpy as np
import torch

import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy
import statistics
import math
import scipy.stats as ss
tstart=time.time()
from config import set_args

########################################################################################################################


args = set_args()
args.output='./res/'+args.experiment+'_'+args.approach+'_'+str(args.note)+'.txt'


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
if 'cat_ncl' in args.approach:
    from approaches import cat_ncl as approach

if 'mlp_cat' in args.approach:
    from networks import mlp_cat as network
elif 'cnn_cat' in args.approach:
    from networks import cnn_cat as network
########################################################################################################################

# Load
data,taskcla,inputsize=dataloader.get(seed=args.seed,args=args)
print('Input size =',inputsize,'\nTask info =',taskcla)

print('taskcla: ',len(taskcla))


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


def auto_similarity(t,appr):
    '''
    use this to detect similarity by transfer and reference network
    Args:
        appr: apprpach
        t: current task id
    Returns: similarity
    '''


    if t > 0:


        for pre_task in range(t+1):
            print('pre_task: ',pre_task)
            print('t: ',t)
            pre_task_torch = torch.autograd.Variable(torch.LongTensor([pre_task]).cuda(),volatile=False)

            # gfc1,gfc2 = appr.model.mask(pre_task_torch,s=appr.smax)
            gfc1,gfc2 = appr.model.mask(pre_task_torch)

            gfc1=gfc1.detach()
            gfc2=gfc2.detach()
            pre_mask=[gfc1,gfc2]

            if pre_task == t: # the last one

                print('>>> Now Training Phase: {:6s} <<<'.format('reference'))
                appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='reference',args=args,
                                            pre_mask=pre_mask,pre_task=pre_task) # implemented as random mask
            elif pre_task != t:
                print('>>> Now Training Phase: {:6s} <<<'.format('transfer'))
                appr.train(t,xtrain,ytrain,xvalid,yvalid,phase='transfer',args=args,
                                            pre_mask=pre_mask,pre_task=pre_task)

            if pre_task == t: # the last one
                test_loss,test_acc=appr.eval(t,xvalid,yvalid,phase='reference',
                                             pre_mask=pre_mask,pre_task=pre_task)
            elif pre_task != t:
                test_loss,test_acc=appr.eval(t,xvalid,yvalid,phase='transfer',
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


    similarity = [0]
    if t > 0:
        acc_list = acc_transfer[t][:t] #t from 0
        print('acc_list: ',acc_list)

        if 'auto' in args.similarity_detection:
            similarity = [0 if (acc_list[acc_id] <= acc_transfer[t][t]) else 1 for acc_id in range(len(acc_list))] # remove all acc < 0.5
        else:
            raise NotImplementedError


        for source_task in range(len(similarity)):
            similarity_transfer[t,source_task]=similarity[source_task]
        print('Save at similarity_transfer')
        np.savetxt(args.output + '_similarity_transfer',similarity_transfer,'%.4f',delimiter='\t')


    print('similarity: ',similarity)
    return similarity

def read_pre_computed_similarity(f_name):
    '''
    use this if you save the computed similarity or you prefere predefine similarity
    Args:
        f_name: similarity file name (should be a matrix in .txt)
    Returns: similarity

    '''
    with open(f_name,'r') as f:
        similarity = [int(float(_))for _ in f.readlines()[t].split('\t')[:t]]
    return similarity


def true_similarity(t,data):
    '''
    mannually set similarity (federated datasets are set to similar, while all others are dissimilar)
    Args:
        data: data
        t: current task id
    Returns: similarity
    '''

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



def all_one_similarity(t,data):
    '''
    all set to similar
    Args:
        data: data
        t: current task id
    Returns: similarity
    '''

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
    return similarity

def all_zero_similarity(t,data):
    '''
    all set to dissimilar
    Args:
        data: data
        t: current task id
    Returns: similarity
    '''

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



# Inits
print('Inits...')
net=network.Net(inputsize,taskcla,nhid=args.nhid,args=args).cuda()
utils.print_model_report(net)

appr=approach.Appr(net,args=args)

print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

check_federated = approach.CheckFederated()

similarity = [0]
history_mask_back = []
history_mask_pre = []
similarities = []

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
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t


    candidate_phases = ['mcl']

    for candidate_phase in candidate_phases:
        if candidate_phase == 'mcl' and 'auto' in args.similarity_detection:
            similarity = auto_similarity(task,appr)
        elif candidate_phase == 'mcl' and 'by-name' in args.similarity_detection:
            similarity = true_similarity(task,data)
        elif candidate_phase == 'mcl' and 'all-one' in args.similarity_detection:
            similarity = all_one_similarity(task,data)
        elif candidate_phase == 'mcl' and 'all-zero' in args.similarity_detection:
            similarity = all_zero_similarity(task,data)

        # else:
        #     raise NotImplementedError

        similarities.append(similarity)
        check_federated.set_similarities(similarities)


        print('>>> Now Training Phase: {:6s} <<<'.format(candidate_phase))


        appr.train(task,xtrain,ytrain,xvalid,yvalid,candidate_phase,args,
                   similarity=similarity,history_mask_back=history_mask_back,
                   history_mask_pre=history_mask_pre,check_federated=check_federated)

        print('-'*100)

        if candidate_phase == 'mcl':
            history_mask_back.append(dict((k, v.data.clone()) for k, v in appr.mask_back.items()) )
            history_mask_pre.append([m.data.clone() for m in appr.mask_pre])

            for u in range(t+1):
                xtest=data[u]['test']['x'].cuda()
                ytest=data[u]['test']['y'].cuda()

                xvalid=data[u]['valid']['x'].cuda()
                yvalid=data[u]['valid']['y'].cuda()

                test_loss,test_acc=appr.test(u,xtest,ytest,candidate_phase,args,
                                             similarity=similarity,history_mask_pre=history_mask_pre,
                                             check_federated=check_federated,
                                             xvalid=xvalid,yvalid=yvalid)
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



with open(performance_output_mcl_backward,'w') as file_backward, open(performance_output_mcl_forward,'w') as file_forward:
    for j in range(acc_mcl.shape[1]):
        file_backward.writelines(str(acc_mcl[-1][j]) + '\n')

    for j in range(acc_mcl.shape[1]):
        file_forward.writelines(str(acc_mcl[j][j]) + '\n')



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