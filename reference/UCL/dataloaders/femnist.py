import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0.10,tasknum = 10,args=0):
    data={}
    taskcla=[]
    size=[1,28,28]


    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}

    if 'small' in args.note:
        data_type = 'small'
    elif 'full' in args.note:
        data_type = 'full'

    train_dataset = FEMMNISTTrain(root_dir='/home/zixuan/KAN/image/dat/femnist/'+data_type+'/iid/train/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['train'] = train_dataset

    test_dataset = FEMMNISTTest(root_dir='/home/zixuan/KAN/image/dat/femnist/'+data_type+'/iid/test/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test'] = test_dataset




    # number of example
    # x = FEMMNISTTrain(root_dir='/home/zixuan/KAN/image/dat/femnist/'+data_type+'/iid/train/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    # y=torch.utils.data.DataLoader(x,batch_size=1,shuffle=True)
    # print(len([0 for user, image, target in y]))
    #
    # x = FEMMNISTTest(root_dir='/home/zixuan/KAN/image/dat/femnist/'+data_type+'/iid/test/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    # y=torch.utils.data.DataLoader(x,batch_size=1,shuffle=True)
    # print(len([0 for user, image, target in y]))
    #
    # number of example

    #
    users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True)])]
    users.sort()
    print('users: ',users)
    print('users length: ',len(users))
    # # totally 47 classes, each tasks 5 classes
    #
    for task_id,user in enumerate(users):
        data[task_id]={}
        data[task_id]['name'] = 'femnist-'+str(user)
        data[task_id]['ncla'] = 62


    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

        for task_id,user in enumerate(users):
            data[task_id][s]={'x': [],'y': []}

        for user,image,target in loader:
            label=target.numpy()[0]

            data[users.index(user[0])][s]['x'].append(image)
            data[users.index(user[0])][s]['y'].append(label)

    # # "Unify" and save
    for n,user in enumerate(users):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    # for t in data.keys():
    #     data[t]['valid'] = {}
    #     data[t]['valid']['x'] = data[t]['train']['x'].clone()
    #     data[t]['valid']['y'] = data[t]['train']['y'].clone()

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

    # # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    #
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

