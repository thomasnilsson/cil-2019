import os
import re
import sys
import pickle
import h5py
import numpy as np 
from datetime import datetime
from sklearn.model_selection import train_test_split

from keras_self_attention import SeqSelfAttention

# Tensorflow modules
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

# Local modules
import rnn_models
import constants
import submit

def predict_validation(rnn, x_val, y_val):
    print("Predicting the validation set...")
    y_pred = rnn.predict(x_val, batch_size=2048, verbose=1)
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    y_pred = y_pred.ravel().astype(int)
    val_acc = (y_val == y_pred).mean()
    print("Validation accuracy:", val_acc)

def predict_test(rnn, x_test, model_name):
    # Predict on test data and make submission
    print("Predicting the test set...")
    y_pred = rnn.predict(x_test, verbose=1, batch_size=2048)
    submit.generate_submission_file(y_pred, model_name)

def load_and_evaluate_rnn(args):
    complete_path = constants.SAVED_RNN_MODELS_DIR + args.model_file
    print("EVALUATING MODEL %s..." % args.model_file)
    rnn = load_model(complete_path, custom_objects=SeqSelfAttention.get_custom_objects())
    _, _, x_val, _, y_val, _ = get_rnn_data(args)
    predict_validation(rnn, x_val, y_val)

def get_rnn_data(args):
    # LOAD DATASET
    if args.use_full_dataset:
        id_dataset = pickle.load(open(constants.ID_DATASET_FULL_PATH, "rb"))
        data_index = pickle.load(open(constants.DATA_INDEX_FULL_PATH, "rb"))
    else:
        id_dataset = pickle.load(open(constants.ID_DATASET_SMALL_PATH, "rb"))
        data_index = pickle.load(open(constants.DATA_INDEX_SMALL_PATH, "rb"))
    
    glove_embeddings = pickle.load(open(constants.EMBEDDINGS_DIR + "stanford_20k.WordEmbedding", "rb"))

    # Split dataset 90%/10%, and shuffle
    x_train = id_dataset["train_tweets"][data_index["train_index"]]
    x_val = id_dataset["train_tweets"][data_index["test_index"]]
    y_train = id_dataset["train_labels"][data_index["train_index"]]
    y_val = id_dataset["train_labels"][data_index["test_index"]]
    x_test = id_dataset["test_tweets"]

    return glove_embeddings.vectors, x_train, x_val, y_train, y_val, x_test
            

def train_rnn(args):
    # Load dataset and embeddings
    embedding_matrix, x_train, x_val, y_train, y_val, x_test = get_rnn_data(args)

    # RNN parameters
    params = {"embedding_matrix" : embedding_matrix, "cell_size" : args.cell_size}
    
    # CREATE RNN MODEL
    if args.model_type == "rnn_simple":
        rnn = rnn_models.simple_model(**params)
    elif args.model_type == "rnn_conv":
        rnn = rnn_models.conv_model(**params)
    elif args.model_type == "rnn_pooling":
        rnn = rnn_models.pooling_model(**params)
    elif args.model_type == "rnn_attention":
        rnn = rnn_models.attention_model(**params)
    elif args.model_type == "rnn_pooled_attention":
        rnn = rnn_models.pooled_attention_model(**params)
    
    # Show the model summary
    print(rnn.summary())
    
    # TRAIN MODEL
    timestamp =  str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_name = args.model_type + "_" + timestamp
    model_file = os.path.join(constants.SAVED_RNN_MODELS_DIR, model_name + ".hdf5")
    
    print("Saving model to:", model_file)
    checkpoint = ModelCheckpoint(model_file,
                                monitor='val_acc', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='max')

    history = rnn.fit(x=x_train, 
                    y=y_train, 
                    validation_data=(x_val, y_val), 
                    batch_size=args.batch_size, 
                    callbacks=[checkpoint], 
                    epochs=args.epochs, 
                    verbose=1)

    # Save losses
    history_dict = history.history
    pickle.dump(history_dict, open(constants.SAVED_RNN_MODELS_DIR + "history_" + model_name + ".dict", "wb"))

    predict_test(rnn, x_test, model_name)
