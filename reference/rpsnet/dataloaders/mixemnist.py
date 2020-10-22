import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from collections import Counter

def get(seed=0, fixed_order=False, pc_valid=0, args=0):
    size=[1,28,28]

    data = {}
    taskcla = []

    data_emnist, taskcla_emnist, size_emnist = read_emnist(seed=seed,args=args)
    data_femnist, taskcla_femnist, size_femnist = read_femnist(seed=seed,args=args)

    if 'random' in args.note:

        all_emnist = [data_emnist[x]['name'] for x in range(args.dis_ntasks)]
        all_femnist = [data_femnist[x]['name'] for x in range(args.sim_ntasks)]

        f_name = 'mixemnist_random_'+str(args.dis_ntasks+args.sim_ntasks)

        with open(f_name,'r') as f_random_seq:
            random_sep = f_random_seq.readlines()[args.idrandom].split()

        print(random_sep)
        for task_id in range(args.dis_ntasks+args.sim_ntasks):
            if 'emnist' in random_sep[task_id]:# Even
                emnist_id = all_emnist.index(random_sep[task_id])
                data[task_id] = data_emnist[emnist_id]
                taskcla.append((task_id,data_emnist[emnist_id]['ncla']))

            elif 'fe-mnist'in random_sep[task_id]:
                femnist_id = all_femnist.index(random_sep[task_id])
                data[task_id] = data_femnist[femnist_id]
                taskcla.append((task_id,data_femnist[femnist_id]['ncla']))

    # print(stack_id)
    print(taskcla)
    return data,taskcla,size

def read_emnist(seed=0,fixed_order=False,pc_valid=0,args=0):

    print('Read MNIST')

    data={}
    taskcla=[]
    size=[1,28,28]
    n_tasks = args.dis_ntasks
    class_per_task = args.classptask
    n_class = 47
    counter = {}

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.EMNIST('./dat/',train=True,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.EMNIST('./dat/',train=False,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))


    for task_id in range(n_tasks):
        if task_id < n_tasks-1:
            data[task_id]={}
            data[task_id]['name'] = 'emnist-'+str(task_id*class_per_task)+'-'+str(task_id*class_per_task+class_per_task)
            data[task_id]['ncla'] = class_per_task

        elif task_id == n_tasks-1: #last task
            data[task_id]={}
            uncovered_class = n_class - class_per_task*task_id
            data[task_id]['name'] = 'emnist-'+str(task_id*class_per_task)+'-'+str(task_id*class_per_task+uncovered_class)
            data[task_id]['ncla'] = uncovered_class


    training_c = 0
    testing_c = 0

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

            if label//class_per_task in data:
                data[label//class_per_task][s]['x'].append(image)
                data[label//class_per_task][s]['y'].append(label%class_per_task)

            elif label//class_per_task not in data: # put to last one
                data[n_tasks-1][s]['x'].append(image)
                data[n_tasks-1][s]['y'].append(label%uncovered_class)

            if 'train' in s:
                training_c+= 1
                # if training_c> 50000:
                if training_c>=6000:
                    break

            if 'test' in s:
                testing_c+= 1
                # if training_c> 10000:
                if testing_c>=800:
                    break


    print('testing_c: ',testing_c)
    print('training len: ',len(data[n_tasks-1]['train']['x']))
    print('testing len: ',len(data[n_tasks-1]['test']['x']))


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
        # print('TSA: ',set(data[t]['valid']['y']))
    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

def read_femnist(seed=0,fixed_order=False,pc_valid=0.10,remain=0,args=0):

    print('Read FEMNIST')
    data={}
    taskcla=[]
    size=[1,28,28]
    n_tasks = args.sim_ntasks
    class_per_task = args.classptask

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}

    if 'small' in args.note:
        data_type = 'small'
    elif 'full' in args.note:
        data_type = 'full'


    if args.sim_ntasks==10:
        train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset


    else:
        train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset


    users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True)])]
    users.sort()
    print('users: ',users)
    print('users length: ',len(users))
    # # totally 47 classes, each tasks 5 classes
    #
    for task_id,user in enumerate(users):
        data[task_id]={}
        data[task_id]['name'] = 'fe-mnist-'+str(user)
        data[task_id]['ncla'] = args.num_class_femnist

    training_c = 0
    testing_c = 0

    for s in ['train','test']:
        print('s: ',s)
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

        for task_id,user in enumerate(users):
            data[task_id][s]={'x': [],'y': []}

        count_label = []
        for user,image,target in loader:

            label=target.numpy()[0]

            if label > args.num_class_femnist-1:
               continue

            data[users.index(user[0])][s]['x'].append(image)
            data[users.index(user[0])][s]['y'].append(label)
            count_label.append(label)

        print('count: ',Counter(count_label))

        # print('testing_c: ',testing_c)
    print('training len: ',sum([len(value['train']['x']) for key, value in data.items()]))
    print('testing len: ',sum([len(value['test']['x']) for key, value in data.items()]))


    # # "Unify" and save
    for n,user in enumerate(users):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()


    return data,taskcla,size






########################################################################################################################

# customize dataset class

class FEMMNISTTrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.size=[1,28,28]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            self.x.append(torch.from_numpy(np.array(data)))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        #number of class
        print(len(set([b for a in self.y for b in a])))
        #number of class

        self.x=torch.cat(self.x,0).view(-1,self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y






class FEMMNISTTest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.size=[1,28,28]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            self.x.append(torch.from_numpy(np.array(data)))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y

