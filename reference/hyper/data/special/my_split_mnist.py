#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :split_mnist.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :04/11/2019
# @version         :1.0
# @python_version  :3.6.7
"""
Split MNIST Dataset
^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.split_mnist` contains a wrapper for data
handlers for the SplitMNIST task.
"""
import numpy as np

from data.mnist_data import MNISTData
import torch



def get_split_MNIST_handlers(data_path, use_one_hot=True,data=0,config=0):
    """This method instantiates 5 objects of the class :class:`SplitMNIST` which
    will contain a disjoint set of labels.

    The SplitMNIST task consists of 5 tasks corresponding to the images with
    labels [0,1], [2,3], [4,5], [6,7], [8,9].

    Args:
        data_path: Where should the MNIST dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: Whether the class labels should be represented in a one-hot
            encoding.
        validation_size: The size of the validation set of each individual
            data handler.
        steps: Number of classes to put into one data handler. If default
            every data handler will include 2 digits, otherwise 1.
    Returns:
        A list of data handlers, each corresponding to a :class:`SplitMNIST`
        object,
    """
    print('Creating data handlers for SplitMNIST tasks ...')
    out_shape = [dim[1] for dim in config.taskcla]

    handlers = []
    for i in range(0, config.num_tasks):
        handlers.append(SplitMNIST(data_path, use_one_hot=use_one_hot,data=data[i],
                                   out_shape=out_shape[i],config=config))
        print('done for task ' + str(i))

    print('Creating data handlers for SplitMNIST tasks ... Done')

    return handlers





class SplitMNIST(MNISTData):
    """An instance of the class shall represent a SplitMNIST task.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: Whether the class labels should be represented in a
            one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        labels: The labels that should be part of this task.
        full_out_dim: Choose the original MNIST instead of the the new
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
    """
    def __init__(self, data_path, use_one_hot=False,full_out_dim=False,data=0,config=0,out_shape=0):
        super().__init__(data_path, use_one_hot=use_one_hot)

        self._data['num_classes'] = config.ntasks



        self._data['train_inds'] = np.arange(data['train']['x'].size(0))
        self._data['val_inds'] = np.arange(data['train']['x'].size(0),data['train']['x'].size(0)+data['valid']['x'].size(0))
        self._data['test_inds'] = np.arange(data['train']['x'].size(0)+data['valid']['x'].size(0),
                                           data['train']['x'].size(0)+data['valid']['x'].size(0)+data['test']['x'].size(0))

        train_data = data['train']['x'].view(-1,28*28*1)
        valid_data = data['valid']['x'].view(-1,28*28*1)
        test_data = data['test']['x'].view(-1,28*28*1)

        if self._data['is_one_hot']:
            nb_digits = out_shape


            train_size = data['train']['y'].size(0)
            train_labels = torch.FloatTensor(train_size, nb_digits)
            train_labels.zero_()
            train_labels.scatter_(1, data['train']['y'].view(-1,1), 1)

            valid_size = data['valid']['y'].size(0)
            valid_labels = torch.FloatTensor(valid_size, nb_digits)
            valid_labels.zero_()
            valid_labels.scatter_(1, data['valid']['y'].view(-1,1), 1)

            test_size = data['test']['y'].size(0)
            test_labels = torch.FloatTensor(test_size, nb_digits)
            test_labels.zero_()
            test_labels.scatter_(1, data['test']['y'].view(-1,1), 1)

        if not full_out_dim:
            # Note, we may also have to adapt the output shape appropriately.
            if self.is_one_hot:
                self._data['out_shape'] = out_shape


        # self._data['train_inds'] = self._data['test_inds']

        self._data['in_data'] = torch.cat([train_data,valid_data,test_data]).cpu().numpy()
        self._data['out_data'] = torch.cat([train_labels,valid_labels,test_labels]).cpu().numpy()

        n_val = self._data['val_inds'].size




        print("self._data['train_inds']",self._data['train_inds'].shape)
        print("self._data['test_inds']",self._data['test_inds'].shape)
        print("self._data['val_inds']",self._data['val_inds'].shape)
        print("self.num_train_samples",self.num_train_samples)
        print("self._data['in_data']",self._data['in_data'].shape)
        print("self._data['out_data']",self._data['out_data'].shape)
        print("self.num_train_samples",self.num_train_samples)
        print("self._data['train_inds']",self._data['train_inds'])
        print("self._data['test_inds']",self._data['test_inds'])
        print("self._data['val_inds']",self._data['val_inds'])
        print("self._data['out_data']",self._data['out_data'])
        print("self._data['in_data']",self._data['in_data'])


        print('Created EMNIST task with and %d train, %d test '
              % (self._data['train_inds'].size, self._data['test_inds'].size) +
              'and %d val samples.' % (n_val))

    def transform_outputs(self, outputs):
        """Transform the outputs from the 10D MNIST dataset into proper 2D
        labels.

        Example:
            Split with labels [2,3]

            1-hot encodings: [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            labels: 3 -> 1

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        labels = self._labels
        if self.is_one_hot:
            assert(outputs.shape[1] == self._data['num_classes'])
            mask = np.zeros(self._data['num_classes'], dtype=np.bool)
            mask[labels] = True

            return outputs[:, mask]
        else:
            assert (outputs.shape[1] == 1)
            ret = outputs.copy()
            for i, l in enumerate(labels):
                ret[ret == l] = i
            return ret

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitMNIST'

if __name__ == '__main__':
    pass
