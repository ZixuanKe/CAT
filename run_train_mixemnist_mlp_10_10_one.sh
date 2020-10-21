#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 3
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
	--experiment mixemnist \
	--approach mlp_one \
	--note random$id,ntasks20 \
	--dis_ntasks 10 \
	--classptask 5 \
	--idrandom $id \
	--data_size small
done