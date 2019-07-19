import os
import re
import sys
import pickle
import h5py
import constants

from tqdm import tqdm

import numpy as np 
import keras

from keras.models import Model
from keras.models import Sequential

# Custom package
from keras_self_attention import SeqSelfAttention

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import submit

def perform_prediction(model, x):
        y = model.predict(x, batch_size=2048, verbose=1)
        y[y <= 0.5] = 0
        y[y > 0.5] = 1
        y = y.ravel().astype(int)
        return y

def calc_val_acc(y, target):
    y[y <= 0.5] = 0
    y[y > 0.5] = 1
    y = y.ravel().astype(int)
    matching = (y == target).astype(int)
    print("Validation Acc:", matching.mean())

def maj_vote(prediction_matrix):
    # The bincount function requires non-negative entries
    prediction_matrix[prediction_matrix == -1] = 0 
    return np.array([np.bincount(row).argmax() for row in prediction_matrix])

def run(args):
    # LOAD DATASET
    if args.use_full_dataset:
        id_dataset = pickle.load(open(constants.ID_DATASET_FULL_PATH, "rb"))
        data_index = pickle.load(open(constants.DATA_INDEX_FULL_PATH, "rb"))
    else:
        id_dataset = pickle.load(open(constants.ID_DATASET_SMALL_PATH, "rb"))
        data_index = pickle.load(open(constants.DATA_INDEX_SMALL_PATH, "rb"))
    
    glove_embeddings = pickle.load(open("../embeddings/stanford_20k.WordEmbedding", "rb"))
    external_embeddings = glove_embeddings.vectors

    # Split dataset 90%/10%, and shuffle
    x_val = id_dataset["train_tweets"][data_index["test_index"]]
    y_val = id_dataset["train_labels"][data_index["test_index"]]
    x_test = id_dataset["test_tweets"]    
    
    val_prediction_matrix_path = constants.SAVED_RNN_MODELS_DIR + "val_prediction_matrix.obj"

    # Specify model
    model_files = [
        "rnn_pooling_256.hdf5",
        "rnn_attention_256.hdf5",
        "rnn_conv_256.hdf5",
        "rnn_pooled_attention_512.hdf5",
        "rnn_simple_256.hdf5",
    ]
    # Load specified models
    loaded_models = [load_model(os.path.join(constants.SAVED_RNN_MODELS_DIR, f), 
                    custom_objects=SeqSelfAttention.get_custom_objects()) 
                    for f in tqdm(model_files)]
    
    if os.path.isfile(val_prediction_matrix_path):
        print("Found prediction matrix at %s, loading it..." % val_prediction_matrix_path)
        val_prediction_matrix = pickle.load(open(val_prediction_matrix_path, "rb"))
    else:
        print("Did not find prediction matrix at %s, creating it from scratch..." % val_prediction_matrix_path)
        # Predict on x_val
        val_prediction_matrix = np.array([perform_prediction(m, x_val) for m in tqdm(loaded_models)]).T
        pickle.dump(val_prediction_matrix, val_prediction_matrix_path, "wb")

    
    # Calc majority votes
    val_majority_votes = maj_vote(val_prediction_matrix)

    # Calculate validation accuracy
    calc_val_acc(val_majority_votes, y_val)

    # Predict on test set
    print("Predicting on the test set...")
    test_prediction_matrix = np.array([perform_prediction(m, x_test) for m in tqdm(loaded_models)]).T
    test_majority_votes = maj_vote(test_prediction_matrix)

    submit.generate_submission_file(test_majority_votes, "majority")