import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
import scipy.io
########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0,remain=0):
    data={}
    taskcla=[]
    size=[1,9,1]
    n_tasks = 29

    # MNIST

    mat_feature = scipy.io.loadmat('./dat/LandmineData/LandmineData.mat')['feature'][0]
    mat_label = scipy.io.loadmat('./dat/LandmineData/LandmineData.mat')['label'][0]

    for task_id,task_data in enumerate(mat_feature):
        data[task_id]={}
        data[task_id]['name'] = 'landmine-'+str(task_id)
        data[task_id]['ncla'] = 2

        data[task_id]['train']={'x': [],'y': []}
        data[task_id]['valid']={'x': [],'y': []}
        data[task_id]['test']={'x': [],'y': []}

    for task_id,task_data in enumerate(mat_feature):
        num_example = len(task_data)

        print(len([1 for x in mat_label[task_id] if x == 0]))
        print(len([1 for x in mat_label[task_id] if x == 1]))

        if remain!=0:
            reduce_index = np.random.choice(range(int(num_example*0.8)), remain, replace=False)
            x_train = torch.from_numpy(task_data[reduce_index]).float()
            y_train = mat_label[task_id][reduce_index]

        elif remain==0:
            x_train = torch.from_numpy(task_data[:int(num_example*0.8)]).float()
            y_train = mat_label[task_id][:int(num_example*0.8)]

        x_develop = torch.from_numpy(task_data[int(num_example*0.8):int(num_example*0.9)]).float()
        x_test = torch.from_numpy(task_data[int(num_example*0.9):]).float()

        y_develop = mat_label[task_id][int(num_example*0.8):int(num_example*0.9)]
        y_test = mat_label[task_id][int(num_example*0.9):]

        data[task_id]['train']['x'].append(x_train)
        data[task_id]['valid']['x'].append(x_develop)
        data[task_id]['test']['x'].append(x_test)

        data[task_id]['train']['y'].append(y_train)
        data[task_id]['valid']['y'].append(y_develop)
        data[task_id]['test']['y'].append(y_test)

    # "Unify" and save
    for n in range(n_tasks):
        for s in ['train','valid','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.from_numpy(np.array(data[n][s]['y'],dtype=int)).view(-1)


    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
