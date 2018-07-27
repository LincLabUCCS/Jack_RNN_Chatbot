#!/bin/bash

EPOCHS=1
BATCH_SIZE=50 #default=40

# MODEL='original'

## declare an array variable
declare -a arr=('original' 'hinge' 'softx' 'sigmx' 'softxl' 'sigmxl' 'wxe')

## now loop through the above array
for MODEL in "${arr[@]}"
do
	>&2 echo $MODEL 
	time python train.py --data_dir data/scotus --save_dir models/$MODEL --training_loss $MODEL --batch_size $BATCH_SIZE --num_epochs=$EPOCHS
done

