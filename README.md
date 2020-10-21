



# CAT (Continual learning with forgetting Avoidance and knowledge Transfer)

This repository contains the code for our NeurIPS 2020 Paper **"Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks"** by Zixuan Ke, Bing Liu and Xingchang Huang.

## Abstract
Existing research on continual learning of a sequence of tasks focused on dealing  with _catastrophic forgetting_, where the tasks are assumed to be dissimilar and have  little shared knowledge. Some work has also been done to transfer previously learned knowledge to the new task when the tasks are similar and have shared   knowledge. To the best of our knowledge, no technique has been proposed to learn a sequence of mixed similar and dissimilar tasks that can deal with forgetting and also transfer knowledge forward and backward. This paper proposes such a technique to learn both types of tasks in the same network. For dissimilar tasks, the algorithm focuses on dealing with forgetting, and for similar tasks, the algorithm focuses on selectively transferring the knowledge learned from some similar previous tasks to improve the new task learning. Additionally, the algorithm automatically detects whether a new task is similar to any previous tasks. Empirical evaluation using sequences of mixed tasks demonstrates the effectiveness of the proposed model

## Files
**/res**: all results saved in this folder  
**/dat**: processed data  
**/dataloader**: contained dataloader for mixed sequences of (EMNIST, F-EMNIST) and (CIFAR100, F-CelebA)  
**/reference**: additional code for baselines (specifically UCL, HyperNet and RPSNet)  
**/approaches**: code for training  
**/networks**: code for network architecture  

## Installing
**run_train_mixemnist_mlp_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with NCL  
**run_train_mixemnist_mlp_hat_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with HAT  
**run_train_mixemnist_mlp_hat_10_10_ncl.sh**: Run M(EMNIST-10, F-EMNIST) with ONE  
**run_train_mixemnist_mlp_remove_att_10_10_ncl** Run M(EMNIST-10, F-EMNIST) with CAT without attention  
**run_train_mixemnist_mlp_10_20_ncl.sh**: Run M(EMNIST-20, F-EMNIST) with NCL  

## Datasets
**EMNIST**: Automatically download when the code is run  
**CIFAR100**: Automatically download when the code is run  
**Federated-CelebA**: Please download from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and follow the instruction  of [Leaf](https://github.com/TalwalkarLab/leaf).  Processed files are in **/dat** folder  
**Federated-EMNIST**: Generated from [Leaf](https://github.com/TalwalkarLab/leaf).  Processed files are in **/dat** folder  

## Acknowledgments
 [HAT](https://github.com/joansj/hat/tree/master/src)  
 [RPSNet](https://github.com/brjathu/RPSnet/blob/master/mnist.py)  
 [UCL](https://github.com/csm9493/UCL)  
 [HyperNet](https://github.com/chrhenning/hypercl/blob/master/toy_example/README.md)  
 
## Contact

Please drop an email to [Zixuan Ke](zke4@uic.edu) if you have any questions. 
