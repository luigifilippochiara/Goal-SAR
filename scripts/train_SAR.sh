#!/bin/bash

python main.py \
	--model_name 'SAR' \
	--phase 'train_test' \
	--num_epochs 500 \
	--batch_size 32 \
	--validate_every 1 \
	--learning_rate 0.001 \
	$@
