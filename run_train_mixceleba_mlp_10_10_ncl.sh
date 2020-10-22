#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
	--experiment mixceleba \
	--approach mlp_ncl \
	--note random$id,ntasks20 \
	--dis_ntasks 10 \
	--classptask 10 \
	--idrandom $id \
	--lr 0.025 \
	--data_size small \
done