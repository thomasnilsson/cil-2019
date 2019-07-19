#!/bin/sh
cd ../src
python3 main.py --action=train --model_type=log_reg --use_full_dataset=1 --embedding_type=stanford
cd ../scripts