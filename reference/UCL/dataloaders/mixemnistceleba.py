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

def get(seed=0, fixed_order=False, pc_valid=0, args=0):
    size=[3,28,28]
    # n_tasks = 10
    n_tasks = 5
    data = {}
    taskcla = []


    data_emnist, taskcla_emnist, size_emnist = read_emnist(seed=seed,args=args,size=size)
    data_femnist, taskcla_femnist, size_femnist = read_femnist(seed=seed,args=args,size=size)

    data_cifar100, taskcla_cifar100, size_cifar100 = read_cifar100(seed=seed,args=args,size=size)
    data_celeba, taskcla_celeba, size_celeba = read_celeba(seed=seed,args=args,size=size)

    if 'stack' in args.note:

        for stack_id in range(5):
            data[stack_id] = data_emnist[stack_id]
            taskcla.append(taskcla_emnist[stack_id])


        for stack_id in range(5,10):
            data[stack_id] = data_femnist[stack_id-n_tasks]
            taskcla.append((stack_id,data_femnist[stack_id-n_tasks]['ncla']))

        for stack_id in range(10,15):
            data[stack_id] = data_cifar100[stack_id-n_tasks*2]
            taskcla.append((stack_id,data_cifar100[stack_id-n_tasks*2]['ncla']))

        for stack_id in range(15,20):
            data[stack_id] = data_celeba[stack_id-n_tasks*2]
            taskcla.append((stack_id,data_celeba[stack_id-n_tasks*2]['ncla']))


    if 'alter' in args.note:

        femnist_id = 0
        emnist_id = 0
        celeba_id = 0
        cifar100_id = 0

        # for stack_id in range(20):
        for stack_id in range(10):
            if (stack_id % 2) == 0: # Even
                data[stack_id] = data_emnist[emnist_id]
                taskcla.append((stack_id,data_emnist[emnist_id]['ncla']))
                emnist_id+=1
            else: #odd
                data[stack_id] = data_femnist[femnist_id]
                taskcla.append((stack_id,data_femnist[femnist_id]['ncla']))
                femnist_id+=1

        for stack_id in range(10,20):
            if (stack_id % 2) == 0: # Even
                data[stack_id] = data_cifar100[cifar100_id]
                taskcla.append((stack_id,data_cifar100[cifar100_id]['ncla']))
                cifar100_id+=1
            else: #odd
                data[stack_id] = data_celeba[celeba_id]
                taskcla.append((stack_id,data_celeba[celeba_id]['ncla']))
                celeba_id+=1


    print(taskcla)
    return data,taskcla,size

def read_emnist(seed=0,fixed_order=False,pc_valid=0,args=0,size=0):

    print('Read MNIST')

    data={}
    taskcla=[]
    n_tasks = 10
    counter = {}

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.EMNIST('./dat/',train=True,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.EMNIST('./dat/',train=False,download=True,split='balanced',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

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

            data[label//5][s]['x'].append(image.expand(image.size(0),size[0],size[1],size[2]))
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

def read_femnist(seed=0,fixed_order=False,pc_valid=0.10,args=0,size=0):

    print('Read FEMNIST')
    data={}
    taskcla=[]


    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}

    if 'small-femnist' in args.note:
        data_type = 'small'
    elif 'full-femnist' in args.note:
        data_type = 'full'

    train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['train'] = train_dataset

    test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test'] = test_dataset


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

            data[users.index(user[0])][s]['x'].append(image.expand(image.size(0),size[0],size[1],size[2]))
            data[users.index(user[0])][s]['y'].append(label)

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


def read_cifar100(seed=0,pc_valid=0.10,args=0,size=0):
    data={}
    taskcla=[]
    counter = {}

    if not os.path.isdir('./dat/binary_mixemnistceleba_cifar100/'):
        os.makedirs('./dat/binary_mixemnistceleba_cifar100')

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100('./dat/',train=True,download=True,transform=transforms.Compose([transforms.Resize(size=(size[1],size[2])),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('./dat/',train=False,download=True,transform=transforms.Compose([transforms.Resize(size=(size[1],size[2])),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        n_per_task = len([1 for image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True) if target.numpy()[0] == 1])
        print('n_per_task: ',n_per_task)

        for n in range(10):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:

                n=target.numpy()[0]

                if n in counter:
                    counter[n] += 1
                elif n not in counter:
                    counter[n] = 1

                nn=(n//10)
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n%10)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('./dat/binary_mixemnistceleba_cifar100'),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('./dat/binary_mixemnistceleba_cifar100'),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    ids=list(shuffle(np.arange(10),random_state=seed))
    print('Task order =',ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/binary_mixemnistceleba_cifar100'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/binary_mixemnistceleba_cifar100'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i])

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

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

def read_celeba(seed=0,pc_valid=0.10,args=0,size=0):
    data={}
    taskcla=[]

    num_task = 10

    if 'small-celeba' in args.note:
        data_type = 'small'
    elif 'full-celeba' in args.note:
        data_type = 'full'

    if not os.path.isdir('./dat/'+data_type+'_binary_mixemnistceleba_celeba/'):
        os.makedirs('./dat/'+data_type+'_binary_mixemnistceleba_celeba')

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # celeba
        dat={}
        train_dataset = CELEBATrain(root_dir='./dat/celeba/'+data_type+'/iid/train/',img_dir='./dat/celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(size[1],size[2])),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = CELEBATest(root_dir='./dat/celeba/'+data_type+'/iid/test/',img_dir='./dat/celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(size[1],size[2])),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset

        users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True)])]
        users.sort()
        users = users[:num_task]
        print('users: ',users)
        print('users length: ',len(users))

        # # totally 10 tasks, each tasks 2 classes (whether smiling)
        #
        for task_id,user in enumerate(users):
            data[task_id]={}
            data[task_id]['name'] = 'celeba-'+str(user)
            data[task_id]['ncla'] = 2


        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

            for task_id,user in enumerate(users):
                data[task_id][s]={'x': [],'y': []}

            for user,image,target in loader:
                if user[0] not in users: continue # we dont want too may classes
                label=target.numpy()[0]
                data[users.index(user[0])][s]['x'].append(image)
                data[users.index(user[0])][s]['y'].append(label)

        # # "Unify" and save
        for n,user in enumerate(users):
            for s in ['train','test']:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser('./dat/'+data_type+'_binary_mixemnistceleba_celeba/'),'data'+str(n)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser('./dat/'+data_type+'_binary_mixemnistceleba_celeba/'),'data'+str(n)+s+'y.bin'))

    # number of example
    # need to further slice [:user_num]
    # number of example


    # Load binary files
    data={}
    ids=list(shuffle(np.arange(num_task),random_state=seed))
    print('Task order =',ids)
    for i in range(num_task):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/'+data_type+'_binary_mixemnistceleba_celeba/'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/'+data_type+'_binary_mixemnistceleba_celeba/'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='celeba-'+str(ids[i])


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

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size


########################################################################################################################



# customize dataset class

class CELEBATrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
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






class CELEBATest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
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