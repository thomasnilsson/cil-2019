#!/bin/sh
model_type=${1:-'rnn_simple'}
use_full_dataset=${2:-0}
cell_size=${3:-256}

cd ../src
python3 main.py --model_type=$model_type --epochs=3 --use_full_dataset=$use_full_dataset --cell_size=$cell_size
cd ../scripts