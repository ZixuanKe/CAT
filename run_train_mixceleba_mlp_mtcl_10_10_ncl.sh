#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run_mtcl.py \
	--experiment mixceleba \
	--approach mlp_mtcl_ncl \
	--note random$id,ntasks20 \
	--dis_ntasks 10 \
	--classptask 10 \
	--idrandom $id \
	--lr 0.025 \
	--lr_patience 10 \
	--n_head 5 \
	--data_size small \
	--similarity_detection auto \
	--loss_type multi-loss-joint-Tsim
done