#!/usr/bin/env python3
# Copyright 2019 Johannes Oswald
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
# @title           :train_args.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :12/16/2019
# @version         :1.0
# @python_version  :3.6.8

"""
A collection of helper functions to keep other scripts clean. The main purpose
is to set the hyperparameter values to reproduce results reported in the paper. 
"""

import argparse
from datetime import datetime
from warnings import warn

def _set_default(config):
    """Overwrite default configs.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """
    config.latent_dim = 100
    config.conditional_replay = True
    config.fake_data_full_range = True
    config.show_plots = False
    config.train_class_embeddings = True
    
    if config.experiment == "mlp":
        config = _set_default_split(config)

    config.infer_output_head = False

    if config.replay_method == "gan":
        config = _set_default_gan(config)
        
    return config

def _set_default_split(config):
    """Overwrite default configs for splitMNIST.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """

    # General setup
    config.enc_fc_arch = '400,400'
    config.dec_fc_arch = '400,400'
    config.class_fc_arch = '2000,2000'
    config.enc_lr, config.dec_lr, config.dec_lr_emb = 0.001, 0.001, 0.001
    config.class_lr, config.class_lr_emb = 0.001, 0.001
    config.n_iter = 2000
    config.batch_size = 128
    config.data_dir = '../datasets'
    config.num_tasks = config.tasks_preserve
    config.padding = 0
    config.no_lookahead = False

    # VAE hnet
    config.rp_temb_size = 96
    config.rp_emb_size = 96
    config.rp_hnet_act = "elu"
    config.rp_hyper_chunks = 80000
    config.rp_hnet_arch = '10,10'
    config.rp_beta = 0.01

    # Classifier hnet
    config.class_temb_size = 96
    config.class_emb_size = 96
    config.class_hnet_act = "relu"
    config.class_hyper_chunks = 42000
    config.class_hnet_arch = '10,10'
    config.class_beta = 0.01
    
    #HNET+TIR
    if config.infer_task_id:
        config.hard_targets = True
        config.dec_fc_arch = '50,150'

    #HNET+R
    else:
        config.hard_targets = False
        config.dec_fc_arch = '250,350'
        
    return config


def _set_default_gan(config):
    """Overwrite default configs for GAN training.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """

    #TODO

    return config

if __name__ == '__main__':
    pass


