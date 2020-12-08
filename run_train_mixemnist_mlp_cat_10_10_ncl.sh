#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 3
do
    CUDA_VISIBLE_DEVICES=0 python run_cat.py \
	--experiment mixemnist \
	--approach mlp_cat_ncl \
	--note random$id,ntasks20 \
	--dis_ntasks 10 \
	--classptask 5 \
	--idrandom $id \
	--lr 0.025 \
	--lr_patience 10 \
	--n_head 5 \
	--data_size small \
	--similarity_detection auto \
	--loss_type multi-loss-joint-Tsim
done