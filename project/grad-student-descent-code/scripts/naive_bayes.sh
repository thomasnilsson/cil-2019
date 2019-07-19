#!/bin/sh
cd ../src
python3 main.py --model_type=naive_bayes --use_full_dataset=1
cd ../scripts