import sys
import torch
import torch.nn.functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,voc_size,weights_matrix,args):
        super(Net,self).__init__()

        ncha,width,height=inputsize
        self.taskcla=taskcla
        self.FILTERS = [3, 4, 5]
        self.filters = args.filters
        self.FILTER_NUM=[args.filter_num] * self.filters
        self.WORD_DIM = height
        self.MAX_SENT_LEN = width

        pdrop1=0.2
        pdrop2=0.5

        if args.pdrop1 >= 0:
            pdrop1 = args.pdrop1
        if args.pdrop2 >= 0:
            pdrop2 = args.pdrop2


        self.embedding = torch.nn.Embedding(voc_size, self.WORD_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = False # non trainable

        self.relu=torch.nn.ReLU()

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)

        # self.rnn = torch.nn.GRU(height, height,bidirectional= False,batch_first=True)

        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.fc1=torch.nn.Linear(self.filters * self.FILTER_NUM[0],args.nhid)
        self.fc2=torch.nn.Linear(args.nhid,args.nhid)
        self.fc3=torch.nn.Linear(args.nhid,args.nhid)


        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.nhid,n))

        print('KimNet')
        return

    def forward(self,x):

        h = self.drop1(self.embedding(x).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN))

        h1 = F.max_pool1d(self.drop1(F.relu(self.c1(h))), self.MAX_SENT_LEN - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h=h1.view(x.size(0),-1)

        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        # h=self.drop2(self.relu(self.fc3(h)))



        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y


