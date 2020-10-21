#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
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
# @title           :split_cifar.py
# @author          :jvo
# @contact         :oswald@ini.ethz.ch
# @created         :05/13/2019
# @version         :1.0
# @python_version  :3.7.3
"""
Split CIFAR-10/100 Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.split_cifar` contains a wrapper for data handlers
for the Split-CIFAR10/CIFAR100 task.
"""
# FIXME The code in this module is mostly a copy of the code in the
# corresponding `split_mnist` module.
import numpy as np
import torch
from data import mixceleba
from data import mixemnist
# from data.cifar10_data import CIFAR10Data
from data.cifar100_data import CIFAR100Data

# DELETEME
def get_split_CIFAR_handlers(data_path, use_one_hot=True, validation_size=0,
                             use_data_augmentation=False):
    """Function has been removed. Use :func:`get_split_cifar_handlers` instead.
    """
    raise NotImplementedError('Function has been removed. Use function ' +
                              '"get_split_cifar_handlers" instead.')

def get_split_cifar_handlers(data_path, use_one_hot=True,data=0,config=0):
    """This method will combine 1 object of the class
    :class:`data.cifar10_data.CIFAR10Data` and 5 objects of the class
    :class:`SplitCIFAR100Data`.

    The SplitCIFAR benchmark consists of 6 tasks, corresponding to the images
    in CIFAR-10 and 5 tasks from CIFAR-100 corresponding to the images with
    labels [0-10], [10-20], [20-30], [30-40], [40-50].

    Args:
        data_path: Where should the CIFAR-10 and CIFAR-100 datasets
            be read from? If not existing, the datasets will be downloaded
            into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        validation_size: The size of the validation set of each individual
            data handler.
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
        num_tasks (int): A number between 1 and 11, specifying the number of
            data handlers to be returned. If ``num_tasks=6``, then there will be
            the CIFAR-10 data handler and the first 5 splits of the CIFAR-100
            dataset (as in the usual CIFAR benchmark for CL).

    Returns:
        (list) A list of data handlers. The first being an instance of class
        :class:`data.cifar10_data.CIFAR10Data` and the remaining ones being an
        instance of class :class:`SplitCIFAR100Data`.
    """
    assert(config.ntasks >= 1 and config.ntasks <= 11)
    print('Creating data handlers for SplitCIFAR tasks ...')
    print('My Split Cifar100 ...')

    handlers = []

    for i in range(10):
        handlers.append(SplitCIFAR100Data(data_path,
            use_one_hot=use_one_hot, data=data[i], config=config))
    print('Creating data handlers for SplitCIFAR tasks ... Done')

    return handlers

class SplitCIFAR100Data(CIFAR100Data):
    """An instance of the class shall represent a single SplitCIFAR-100 task.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
            Note, we are using the same data augmentation pipeline as for
            CIFAR-10.
        labels: The labels that should be part of this task.
        full_out_dim: Choose the original CIFAR instead of the the new 
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
    """
    def __init__(self, data_path, use_one_hot=False,full_out_dim=False,data=0,config=0):
        super().__init__(data_path, use_one_hot=use_one_hot)

        ### Overwrite internal data structure. Only keep desired labels.

        # Note, we continue to pretend to be a 100 class problem, such that
        # the user has easy access to the correct labels and has the original
        # 1-hot encodings.



        self._data['train_inds'] = np.arange(data['train']['x'].size(0))
        self._data['val_inds'] = np.arange(data['train']['x'].size(0),data['train']['x'].size(0)+data['valid']['x'].size(0))
        self._data['test_inds'] = np.arange(data['train']['x'].size(0)+data['valid']['x'].size(0),
                                           data['train']['x'].size(0)+data['valid']['x'].size(0)+data['test']['x'].size(0))




        if self._data['is_one_hot']:

            if 'sep-emnist' in config.note and 5 == config.classptask:
                nb_digits = 5
                size = (-1,28*28*1)

            elif 'sep-emnist' in config.note and 2 == config.classptask:
                nb_digits = 7
                size = (-1,28*28*1)

            elif 'mixemnist' in config.note:
                nb_digits=62
                size = (-1,28*28*1)
            elif 'sep-femnist' in config.note:
                nb_digits=62
                size = (-1,28*28*1)

            elif 'sep-cifar100' in config.note:
                nb_digits=config.classptask
                size = (-1,32*32*3)

            elif 'sep-celeba' in config.note:
                nb_digits=2
                size = (-1,32*32*3)

            elif 'mixceleba' in config.note:
                nb_digits=config.classptask
                size = (-1,32*32*3)

            print('nb_digits: ',nb_digits)
            self._data['num_classes'] = nb_digits

            train_data = data['train']['x'].view(size)
            valid_data = data['valid']['x'].view(size)
            test_data = data['test']['x'].view(size)

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
                self._data['out_shape'] = nb_digits


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


        print('Created SplitCIFAR task with and %d train, %d test '
              % (self._data['train_inds'].size, self._data['test_inds'].size) +
              'and %d val samples.' % (n_val))

    def transform_outputs(self, outputs):
        """Transform the outputs from the 100D CIFAR100 dataset
        into proper 10D labels.

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
        return 'SplitCIFAR100'

if __name__ == '__main__':
    pass


