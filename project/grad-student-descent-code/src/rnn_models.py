import os
import re
import sys
import pickle
import h5py
import constants

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

CLIP_LENGTH = 40

def simple_model(embedding_matrix=None, cell_size=256):
    # Set embedding
    vocab_size, embedding_dim = embedding_matrix.shape
    emb = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=CLIP_LENGTH, 
                          weights=[embedding_matrix], 
                          trainable=True)

    model = keras.models.Sequential()
    model.add(emb)
    model.add(GRU(cell_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model

def conv_model(embedding_matrix=None, cell_size=256):
    vocab_size, embedding_dim = embedding_matrix.shape

    # Input layer
    x_input = Input(shape=(CLIP_LENGTH,))
    
    # Embedding layer
    embedded = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=CLIP_LENGTH, 
                          weights=[embedding_matrix], 
                          trainable=True)(x_input)
    
    dropout = SpatialDropout1D(0.3)(embedded)
    
    # RNN layer
    bi_rnn_out = Bidirectional(GRU(cell_size, 
                                dropout=0.2, 
                                recurrent_dropout=0.2))(dropout)
    
    conv_layer = Conv1D(filters=64,
                    kernel_size=2, 
                    padding="valid", 
                    kernel_initializer="he_uniform")(bi_rnn_out)
    
    flatten = Flatten()(conv_layer)
    
    # Output layer
    y_pred = Dense(1, activation="sigmoid")(flatten)
    
    # Compile model
    model = Model(inputs=x_input, outputs=y_pred)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model

def pooling_model(embedding_matrix=None, cell_size=256):
   
    vocab_size, embedding_dim = embedding_matrix.shape
    # Input layer
    x_input = Input(shape=(CLIP_LENGTH,))
    
    # Embedding layer    
    embed = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=CLIP_LENGTH, 
                          weights=[embedding_matrix], 
                          trainable=True)(x_input)
    drop = SpatialDropout1D(0.3)(embed)
    bi_rnn = Bidirectional(GRU(cell_size, return_sequences=True))(drop)
    pool = concatenate([GlobalAveragePooling1D()(bi_rnn), GlobalMaxPooling1D()(bi_rnn)])
    y_pred = Dense(1, activation="sigmoid")(pool)
    
    # Compile model
    model = Model(inputs=x_input, outputs=y_pred)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def attention_model(embedding_matrix=None, cell_size=256):
    vocab_size, embedding_dim = embedding_matrix.shape
    # Input layer
    x_input = Input(shape=(CLIP_LENGTH,))
    
    # Embedding layer
    embed = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=CLIP_LENGTH, 
                          weights=[embedding_matrix], 
                          trainable=True)(x_input)
    drop = SpatialDropout1D(0.3)(embed)
    bi_rnn = Bidirectional(GRU(cell_size, return_sequences=True))(drop)
    att = SeqSelfAttention(attention_activation="sigmoid")(bi_rnn)
    flat = Flatten()(att)
    y_pred = Dense(1, activation="sigmoid")(flat)
    
    # Compile model
    model = Model(inputs=x_input, outputs=y_pred)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def pooled_attention_model(embedding_matrix=None, cell_size=256):
   
    vocab_size, embedding_dim = embedding_matrix.shape
    # Input layer
    x_input = Input(shape=(CLIP_LENGTH,))
    
    # Embedding layer
    embed = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=CLIP_LENGTH, 
                          weights=[embedding_matrix], 
                          trainable=True)(x_input)
    drop = SpatialDropout1D(0.3)(embed)
    bi_rnn = Bidirectional(GRU(cell_size, return_sequences=True))(drop)
    att = SeqSelfAttention(attention_activation="sigmoid")(bi_rnn)
    pool = concatenate([GlobalAveragePooling1D()(att), GlobalMaxPooling1D()(att)])
    y_pred = Dense(1, activation="sigmoid")(pool)
    
    # Compile model
    model = Model(inputs=x_input, outputs=y_pred)
    opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model