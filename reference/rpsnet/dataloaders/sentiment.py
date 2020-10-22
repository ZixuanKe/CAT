import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import pickle
import os
import pymongo
import collections

import numpy as np
from collections import Counter
import random


def get(seed=0,pc_valid=0.10,args=0,max_doc_len=240):
    data={}
    taskcla=[]
    size=[1,240,300]

    f = open('./dat/sentiment/sequenceall','r')
    all_domains = f.readlines()[0].replace('\n','').split()
    all_tasks = len(all_domains)
    if not os.path.isdir('./dat/sentiment/binary_sentiment'):
        os.makedirs('./dat/sentiment/binary_sentiment')

        word2id, weights_matrix, voc_size= compute_embedding(all_domains)

        for n in range(all_tasks):
            data[n]={}
            data[n]['name']='sentiment'
            data[n]['ncla']=2

            train_x, _, train_y = load_inputs_document_mongo2D(
                [all_domains[n]], 'train', word2id, max_doc_len)
            val_x, _, val_y = load_inputs_document_mongo2D(
                [all_domains[n]], 'dev', word2id, max_doc_len)
            test_x, _, test_y = load_inputs_document_mongo2D(
                [all_domains[n]], 'test', word2id,max_doc_len)

            data[n]['train']={'x': train_x,'y': train_y}
            data[n]['valid']={'x': val_x,'y': val_y}
            data[n]['test']={'x': test_x,'y': test_y}


        # "Unify" and save
        for t in data.keys():
            for s in ['train','valid','test']:

                data[t][s]['x']=torch.LongTensor(np.array(data[t][s]['x'],dtype=int))
                data[t][s]['y']=torch.argmax(torch.LongTensor(np.array(data[t][s]['y'],dtype=int)),dim=1).view(-1)

                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('./dat/sentiment/binary_sentiment'),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('./dat/sentiment/binary_sentiment'),'data'+str(t)+s+'y.bin'))

                # print(domain_list[t])
                # print(s)
                # print(data[t][s]['y'].size())

        np.save(os.path.join(os.path.expanduser('./dat/sentiment'),'weights_matrix'), weights_matrix)
        np.save(os.path.join(os.path.expanduser('./dat/sentiment'),'voc_size'),voc_size)


    # Load binary files
    data={}
    ids=list(shuffle(np.arange(all_tasks),random_state=seed))
    print('Task order =',ids)
    for i in range(all_tasks):
        data[i] = dict.fromkeys(['name','ncla','train','test','valid'])
        for s in ['train','valid','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/sentiment/binary_sentiment'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/sentiment/binary_sentiment'),'data'+str(ids[i])+s+'y.bin'))
            #shuffle
            idx = torch.randperm(data[i][s]['x'].size(0))
            data[i][s]['x'] = data[i][s]['x'][idx].view(data[i][s]['x'].size())
            data[i][s]['y'] = data[i][s]['y'][idx].view(data[i][s]['y'].size())

        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']=str(all_domains[ids[i]])



    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n


    weights_matrix = np.load(os.path.join(os.path.expanduser('./dat/sentiment'),'weights_matrix.npy'))
    voc_size= np.load(os.path.join(os.path.expanduser('./dat/sentiment'),'voc_size.npy'))

    data_sentiment={}
    taskcla_sentiment=[]

    all_domains = [data[x]['name'] for x in range(all_tasks)]

    f = open('./dat/sentiment/sequence','r')
    current_domain = f.readlines()[args.idrandom].replace('\n','').split()

    print('current_domain: ',current_domain)
    print('all_domains: ',all_domains)


    for task_id in range(args.ntasks):
        sentiment_id = all_domains.index(current_domain[task_id])
        data_sentiment[task_id] = data[sentiment_id]
        taskcla_sentiment.append((task_id,data[sentiment_id]['ncla']))
        print('data_sentiment: ',data_sentiment[task_id]['name'])
        print('data_sentiment: ',data[sentiment_id]['name'])
        print('sentiment_id: ',sentiment_id)






    return data_sentiment,taskcla_sentiment,size,voc_size,weights_matrix



# facility ===========================


class Vocabulary(object):
    """Vocabulary
    """

    EOS = 'UNK'

    def __init__(self, add_eos=True):
        self._add_eos = add_eos
        self._word_dict = None
        self._word_list = None
        self._voc_size = None

    def load(self, iter_voc_item, word_column='word', index_column='index'):
        """Load an existing vocabulary.

        Args:
            iter_voc_item: Iterable object. This can be a list, a generator or a database cursor.
            word_column (str): Column name that contains the word.
            index_column (str): Column name that contains the word index.

        """
        # load word_dict
        word_dict = dict()
        for doc in iter_voc_item:
            word = doc[word_column]
            index = doc[index_column]
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    def dump(self, word_column='word', index_column='index'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            yield {
                word_column: word,
                index_column: index
            }

    def generate(self, iter_words, words_column='words', min_count=1, verbose_fn=None):
        """Generate a vocabulary from sentences.

        Args:
            iter_words: Iterable object. This can be a list, a generator or a database cursor.
            words_column (str): Column name that contains "words" data.
            min_count (int): Minimum count of the word in the vocabulary.
            verbose_fn ((int) -> None): Verbose function.
                This is useful when iter_words contains much more documents.

        """
        # statistic info
        counter = collections.defaultdict(int)
        for i, doc in enumerate(iter_words, 1):
            words = doc[words_column]
            for word in words:
                counter[word] += 1
            if verbose_fn:
                verbose_fn(i)
        if '' in counter:
            del counter['']

        # generate word_dict (word -> index)
        word_dict = {self.EOS: 0}
        for word, count in counter.items():
            if count < min_count:
                continue
            index = len(word_dict)
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    def indexes_to_words(self, indexes):
        id2word = {}
        for index in range(indexes):
            id2word[index] = self._word_list[index]
        return id2word


class WordEmbedding(object):

    def __init__(self):
        self._word_dict = None
        self._word_list = None
        self._emb_mat = None

    def load(self, iter_emb_item, word_column='word', index_column='index', vector_column='vector'):
        # load word_dict and emb_dict
        word_dict = dict()
        emb_dict = dict()
        for doc in iter_emb_item:
            word = doc[word_column]
            index = doc[index_column]
            vector = doc[vector_column]
            word_dict[word] = index
            emb_dict[index] = vector
        voc_size = len(word_dict)

        # generate word_list
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        # generate emb_list
        emb_list = [None for _ in range(voc_size)]
        for index, vector in emb_dict.items():
            emb_list[index] = vector

        self._word_dict = word_dict
        self._word_list = word_list
        self._emb_mat = np.array(emb_list, np.float32)
        return self

    def dump(self, word_column='word', index_column='index', vector_column='vector'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.
            vector_column (str): Column name for vector.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            vector = self._emb_mat[index]
            yield {
                word_column: word,
                index_column: index,
                vector_column: pickle.dumps(vector)
            }

    def generate(self,
                 voc,
                 iter_pre_trained,
                 word_column='word',
                 vector_column='vector',
                 bound=(-1.0, 1.0),
                 verbose_fn=None):
        """Generate word embedding.

        Args:
            voc (Vocabulary): Vocabulary.
            iter_pre_trained: Iterator/Generator of per-trained word2vec.
            word_column (str): Column name for word.
            vector_column (str): Column name for vector.
            bound (tuple[float]): Bound of the uniform distribution which is used to generate vectors for words that
                not exist in pre-trained word2vec.
            verbose_fn ((int) -> None): Verbose function to indicate progress.

        """
        # inherit input vocabulary's word_dict and word_list
        self._word_dict = voc.word_dict
        self._word_list = voc.word_list

        # generate emb_list
        emb_size = None
        emb_list = [None for _ in range(voc.voc_size)]  # type: list
        for i, doc in enumerate(iter_pre_trained, 1):
            if verbose_fn:
                verbose_fn(i)
            word = doc[word_column]
            vector = doc[vector_column]
            if emb_size is None:
                emb_size = len(vector)
            try:
                index = self._word_dict[word]
            except KeyError:
                continue
            emb_list[index] = vector

        # If a word is not in the pre-trained embeddings, generate a random vector for it
        for i, vector in enumerate(emb_list):
            vector = emb_list[i]
            if vector is None:
                vector = np.random.uniform(bound[0], bound[1], emb_size)
            emb_list[i] = vector

        self._emb_mat = np.array(emb_list, np.float32)
        return self

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    @property
    def emb_mat(self):
        return self._emb_mat


class Label(object):
    def __init__(self):
        self.not_use = ()

    @staticmethod
    def convert_rating_to_POSNEG(ratting):
        if ratting > 3.0:
            return 'POS'
        elif ratting < 3.0:
            return 'NEG'
        else:
            return 'NEU'

    @staticmethod
    def get_label_to_index_dict():
        return {"POS": 1, "NEG": 0}

    @staticmethod
    def get_index_to_label_dict():
        return {1: "POS", 0: "NEG"}

    @staticmethod
    def convert_POSNEG_to_plus1minus1(label):
        if label == "POS" or label == 1:
            return 1
        elif label == "NEG" or label == 0:
            return 0

    @staticmethod
    def convert_plus1minus1_to_POSNEG(label):
        if label == "POS" or label == 1:
            return "POS"
        elif label == "NEG" or label == 0:
            return "NEG"


def load_w2v_mongo(domain_list):
    print('domai_list',domain_list)
    with pymongo.MongoClient() as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM password;')
        db = conn['zixuan_d']
        coll_vocab = db['pn_vocab']

        print('domain_list',domain_list)
        print('Loading vocabulary...')
        voc = Vocabulary()
        voc.load(
            iter_voc_item=(doc for doc in coll_vocab.find({"domain":{"$in":domain_list}})),
            word_column='word',
            index_column='value'
        )
        word2id = voc.word_dict
        print(f'Vocabulary loaded. voc_size={voc.voc_size}')

        print('Generating embeddings...')

        def verbose(i):
            if i % 10000 == 0:
                print(f'Processing {i}', end='\r')

        emb = WordEmbedding()
        emb.generate(
            voc=voc,
            iter_pre_trained=(
                {'word': doc['word'], 'vec': doc['vec']}
                for doc in conn['word2vec']['glove_840B_300d'].find()
            ),
            word_column='word',
            vector_column='vec',
            verbose_fn=verbose
        )
        w2v = emb.emb_mat

    return word2id, w2v, voc.voc_size





def load_y2id_id2y(file):
    y2id = dict()
    id2y = dict()
    with open(file, 'r', encoding='utf-8') as fout:
        for line in fout:
            y, id_y = line.split()
            y2id[y] = int(id_y)
            id2y[int(id_y)] = y
    return y2id, id2y



def load_inputs_document_mongo2D(domains, type_data, word_id_file, max_doc_len, encoding='utf-8'):
    if type(word_id_file) is str:
        word_to_id = load_word2id(word_id_file)
    else:
        word_to_id = word_id_file

    print("domains",domains)

    x, y, sen_len, doc_len = [], [], [], []
    with pymongo.MongoClient() as mongo_client:
        mongo_client['admin'].authenticate('root', 'SELECT * FROM password;')
        db = mongo_client['zixuan_d']
        domain_list = domains

        coll_name = db['pn_' + type_data]
        for domain in domain_list:
            coll_reviews = coll_name.find({"domain": domain})
            for review in coll_reviews:
                label = review['label']
                if label == 0 or label == '0':
                    continue
                t_x = np.zeros((max_doc_len,))  # initialization with zero for padding
                doc = review['lemma']
                doc_flag = False
                doc = ' '.join(doc)

                j = 0  # word j
                words = doc.split()
                for word in words:
                    if j < max_doc_len:
                        if word in word_to_id:
                            t_x[j] = word_to_id[word]
                            j += 1
                        else:
                            t_x[j] = word_to_id['UNK']  # word_to_id['UNK'] = 0
                            j += 1
                    else:
                        break
                if j > 2:  # if more than two words, treading as a sentence
                    doc_flag = True


                # end for sentences
                if doc_flag:
                    doc_len.append(j)  # 'the number of sentences' of each doc in a batch
                    x.append(t_x)
                    # convert_plus1minus1_to_one_zero
                    if label == -1 or label == '-1':
                        label = int(0)
                    y.append(label)
            print('load ' + type_data + ' dataset {} done!'.format(domain))
    y = change_y_to_onehot(y)

    return np.asarray(x), np.asarray(doc_len), np.asarray(y)


def domains():
    domain_list = []
    with pymongo.MongoClient() as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM password;')
        db = conn['zixuan_d']
        coll_domain = db['domain2index']
        for item in coll_domain.find():
            domain_list.append(item['domain'])
    domain_list.sort()
    return domain_list

def compute_embedding(domain):
    word2id, w2v, voc_size = load_w2v_mongo(domain)
    return word2id, w2v, voc_size



def load_word2id(domain):
    with pymongo.MongoClient() as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM password;')
        db = conn['zixuan_d']
        coll_vocab = db['pn_vocab']

        print('Loading {} vocabulary...'.format(domain))
        voc = Vocabulary()
        voc.load(
            iter_voc_item=(doc for doc in coll_vocab.find({"domain": domain})),
            word_column='word',
            index_column='value'
        )
        print(f'Vocabulary loaded. voc_size={voc.voc_size}')
    voc.dump()
    word2id = voc.word_dict
    return word2id



def change_y_to_onehot(y):
    print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)  # the number of classes
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    with open('dat/sentiment/y2id.txt', 'w', encoding='utf-8') as fin:
        for k, v in y_onehot_mapping.items():
            fin.write(str(k) + ' ' + str(v) + '\n')
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1  # only tmp[y_onehot_mapping[label]] = 1, others = 0
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)