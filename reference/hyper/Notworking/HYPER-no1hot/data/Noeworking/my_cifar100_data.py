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
# @title           :data/cifar100_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :05/02/2019
# @version         :1.0
# @python_version  :3.6.8
"""
CIFAR-100 Dataset
-----------------

The module :mod:`data.cifar100_data` contains a handler for the CIFAR 100
dataset.

The dataset consists of 60000 32x32 colour images in 100 classes, with 600
images per class. There are 50000 training images and 10000 test images.

Information about the dataset can be retrieved from:
    https://www.cs.toronto.edu/~kriz/cifar.html
"""
# FIXME: The content of this module is mostly a copy of the module
# 'cifar10_data'. These two should be merged in future.

import os
import numpy as np
import time
import _pickle as pickle
import urllib.request
import tarfile
import matplotlib.pyplot as plt
import sys,argparse
from data.dataset import Dataset
from data.cifar10_data import CIFAR10Data
from data import mixceleba
import torch


class CIFAR100Data(Dataset):
    """An instance of the class shall represent the CIFAR-100 dataset.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        use_data_augmentation (bool, optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor` (hence,
            **only available for PyTorch**, so far).
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
    """
    _DOWNLOAD_PATH = 'https://www.cs.toronto.edu/~kriz/'
    _DOWNLOAD_FILE = 'cifar-100-python.tar.gz'
    _EXTRACTED_FOLDER = 'cifar-100-python'

    _TRAIN_BATCH_FN = 'train'
    _TEST_BATCH_FN = 'test'
    _META_DATA_FN = 'meta'

    def __init__(self,data,use_one_hot=False):
        super().__init__()

        start = time.time()

        print('Reading CIFAR-100 dataset ...')

        # my data: mixceleba; everything for a single task, already split

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 10
        self._data['is_one_hot'] = use_one_hot

        self._data['in_shape'] = [32, 32, 3]
        self._data['out_shape'] = [10 if use_one_hot else 1]

        # Fill the remaining _data fields with the information read from
        # the downloaded files
        # .
        self._read_batches(data)

        # print('dhandlers: ',dhandlers)
        # my data: mixceleba

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def _read_batches(self, data):
        """Read training and testing batch from files.

        The method fills the remaining mandatory fields of the _data attribute,
        that have not been set yet in the constructor.

        The images are converted to match the output shape (32, 32, 3) and
        scaled to have values between 0 and 1. For labels, the correct encoding
        is enforced.

        Args:
            train_fn: Filepath of the train batch.
            test_fn: Filepath of the test batch.
            validation_size: Number of validation samples.
        """

        self._data['train_inds'] = np.arange(data['train']['x'].size(0))
        self._data['val_inds'] = np.arange(data['train']['x'].size(0),data['train']['x'].size(0)+data['valid']['x'].size(0))
        self._data['test_inds'] = np.arange(data['train']['x'].size(0)+data['valid']['x'].size(0),
                                           data['train']['x'].size(0)+data['valid']['x'].size(0)+data['test']['x'].size(0))

        train_data = data['train']['x'].view(-1,32*32*3)
        valid_data = data['valid']['x'].view(-1,32*32*3)
        test_data = data['test']['x'].view(-1,32*32*3)



        if self._data['is_one_hot']:
            train_labels = torch.from_numpy(self._to_one_hot(data['train']['y'].view(-1,1)))
            valid_labels = torch.from_numpy(self._to_one_hot(data['valid']['y'].view(-1,1)))
            test_labels = torch.from_numpy(self._to_one_hot(data['test']['y'].view(-1,1)))


        self._data['in_data'] = torch.cat([train_data,valid_data,test_data]).cpu().numpy()
        self._data['out_data'] = torch.cat([train_labels,valid_labels,test_labels]).cpu().numpy()


        print("self._data['in_data']: ",self._data['in_data'].shape)
        print("self._data['out_data']: ",self._data['out_data'].shape)
        print("self._data['train_inds']: ",len(self._data['train_inds']))
        print("self._data['val_inds']: ",len(self._data['val_inds']))
        print("self._data['test_inds']: ",len(self._data['test_inds']))


        self._augment_inputs = False



    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CIFAR-100'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        The input images are preprocessed if data augmentation is enabled.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if self._augment_inputs and not force_no_preprocessing:
            if mode == 'inference':
                transform = self._test_transform
            elif mode == 'train':
                transform = self._train_transform
            else:
                raise ValueError('"%s" not a valid value for argument "mode".'
                                 % mode)

            return CIFAR10Data.torch_augment_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(self, x, device,
                mode=mode, force_no_preprocessing=force_no_preprocessing)

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CIFAR-100 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._data['cifar100']['fine_label_names'][label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = \
                    self._data['cifar100']['fine_label_names'][pred_label]

                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label) + '\nPrediction: %s (%d)' % \
                             (pred_label_name, pred_label))

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(inputs, self.in_shape)))
        fig.add_subplot(ax)

        if num_inner_plots == 2:
            ax = plt.Subplot(fig, inner_grid[1])
            ax.set_title('Predictions')
            bars = ax.bar(range(self.num_classes), np.squeeze(predictions))
            ax.set_xticks(range(self.num_classes))
            if outputs is not None:
                bars[int(label)].set_color('r')
            fig.add_subplot(ax)
        
    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Re-Implementation of method
        :meth:`data.dataset.Dataset._plot_config`.

        This method has been overriden to ensure, that there are 2 subplots,
        in case the predictions are given.
        """
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)
        
        if predictions is not None and \
                np.shape(predictions)[1] == self.num_classes:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

if __name__ == '__main__':
    pass


