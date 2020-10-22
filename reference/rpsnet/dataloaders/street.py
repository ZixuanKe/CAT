import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
import json
########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0,reduce=0):
    data={}
    taskcla=[]
    size=[1,28,28]
    n_tasks = 10
    counter = {}

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.EMNIST('./dat/',train=True,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.EMNIST('./dat/',train=False,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

    n_per_task = len([1 for image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True) if target.numpy()[0] == 1])
    print('n_per_task: ',n_per_task)
    # totally 47 classes, each tasks 5 classes

    for task_id in range(n_tasks):

        if task_id < 9:
            data[task_id]={}
            data[task_id]['name'] = 'emnist-'+str(task_id*5)+'-'+str(task_id*5+5)
            data[task_id]['ncla'] = 5

        elif task_id == 9: #last task
            data[task_id]={}
            data[task_id]['name'] = 'emnist-'+str(task_id*5)+'-'+str(task_id*5+2)
            data[task_id]['ncla'] = 2

    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

        for task_id in range(n_tasks):
            data[task_id][s]={'x': [],'y': []}

        for image,target in loader:
            label=target.numpy()[0]

            if s == 'train':
                if label in counter:
                    counter[label] += 1
                elif label not in counter:
                    counter[label] = 1

                if counter[label] > n_per_task-reduce: # we do want to reduce some amount of examples
                    continue

            data[label//5][s]['x'].append(image)
            data[label//5][s]['y'].append(label%5)




    # "Unify" and save
    for n in range(n_tasks):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
