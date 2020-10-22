#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
	--experiment mixemnist \
	--approach mlp_pathnet_ncl \
	--note random$id,ntasks30 \
	--dis_ntasks 20 \
	--classptask 2 \
	--idrandom $id \
	--data_size small
done