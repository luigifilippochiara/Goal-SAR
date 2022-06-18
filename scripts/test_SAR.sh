#!/bin/bash

python main.py \
	--model_name 'SAR' \
	--phase 'test' \
	--load_checkpoint 'best' \
	--batch_size 32 \
	$@
