#!/bin/sh
cd ../twitter_data

# Run preprocessing
chmod +x commands.sh
./commands.sh

cd ../src

# Run gradient descent algorithm
python3 train_glove.py --dim=200 --epochs=10
cd ../scripts