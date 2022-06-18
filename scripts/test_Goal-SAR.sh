#!/bin/bash

python main.py \
	--model_name 'Goal_SAR' \
	--phase 'test' \
	--load_checkpoint 'best' \
	--batch_size 32 \
	$@
