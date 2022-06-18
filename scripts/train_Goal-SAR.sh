#!/bin/bash

python main.py \
	--model_name 'Goal_SAR' \
	--phase 'train_test' \
	--num_epochs 300 \
	--batch_size 32 \
	--validate_every 1 \
	--learning_rate 0.0001 \
	$@
