
# 

# CAT (Continual learning with forgetting Avoidance and knowledge Transfer)

This repository contains the code for our  NeuIPS 2020 Paper **"Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks"** by Zixuan Ke, Bing Liu and Xingchang Huang.

## Files

**/res**: all results saved in this folder

**/dat**: processed data

**/dataloader**: contained dataloader for mixed sequences of (EMNIST, F-EMNIST) and (CIFAR100, F-CelebA)

**/reference**: additional code for baselines (specifically UCL, HyperNet and RPSNet)

**/approaches**: code for training

**/networks**: code for network architecture

## How to Run?

**run_train_mixemnist_mlp_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with NCL

**run_train_mixemnist_mlp_hat_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with HAT

**run_train_mixemnist_mlp_hat_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with ONE

**run_train_mixemnist_mlp_remove_att_10_10_ncl** Run M(EMNIST-10, F-EMNIST) with CAT without attention

**run_train_mixemnist_mlp_10_20_ncl.sh**: Run M(EMNIST-20, F-EMNIST) with NCL

**Note that the code will automatically download EMNIST and CIFAR100 to /dat if you don't have them**

## Contact

Please drop an email to [Zixuan Ke](zke4@uic.edu) if you have any questions. 
