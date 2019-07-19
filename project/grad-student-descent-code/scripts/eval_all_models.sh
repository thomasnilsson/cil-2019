#!/bin/sh
cd ../src
# Baseline models
echo "Evaluating LOG_REG_CIL..."
python3 main.py --action=evaluate --embedding_type=cil --use_full_dataset=1 --model_file=log_reg_cil.pickle

echo "Evaluating LOG_REG_STANFORD..."
python3 main.py --action=evaluate --embedding_type=stanford --use_full_dataset=1 --model_file=log_reg_stanford.pickle

echo "Evaluating NAIVE_BAYES..."
python3 main.py --action=evaluate --model_file=naive_bayes --use_full_dataset=1

# RNN models
echo "Evaluating SIMPLE_256_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_simple_256.hdf5

echo "Evaluating CONV_256_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_conv_256.hdf5

echo "Evaluating POOLING_256_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_pooling_256.hdf5

echo "Evaluating ATTENTION_256_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_attention_256.hdf5

echo "Evaluating POOLED_ATTENTION_512_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_pooled_attention_512.hdf5

echo "Evaluating POOLED_ATTENTION_256_RNN..."
python3 main.py --action=evaluate --use_full_dataset=1 --model_file=rnn_pooled_attention_256.hdf5



cd ../scripts