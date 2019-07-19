import os
import numpy as np
import pickle
import random
import gensim
from tqdm import tqdm

# Custom modules
import constants
from word_embedding import WordEmbedding


def load_glove_embedding(glove_type=constants.GLOVE_STANFORD):
    '''
    Loads the glove embedding (stanford or CIL) as
    a WordEmbedding object which has been pickled beforehand.
    '''

    # Check if glove CIL
    if glove_type == constants.GLOVE_CIL:
        path = constants.GLOVE_EMBEDDING_CIL_PATH
        print("Using CIL GloVe embeddings...")
    else:
        path = constants.GLOVE_EMBEDDING_STANFORD_PATH
        print("Using Stanford GloVe embeddings...")
    
    # Load WordEmbedding object
    print("Loading embeddings at %s..." % path)
    embedding = pickle.load(open(path, "rb"))
    return embedding

def _load_word2vec_embedding():
    '''
    Loads the word2vec embedding using a binary file. The
    word2vec embeddings are very large and cannot be pickled
    as a WordEmbedding object.
    '''
    print("Loading word2vec embeddings, this may take a while...")
    w2v_model = gensim\
        .models.KeyedVectors\
        .load_word2vec_format(constants.WORD2VEC_EMBEDDING_PATH, binary=True)
    vocab = {k: i for i,k in enumerate(w2v_model.vocab)}
    return WordEmbedding(w2v_model.vectors, vocab)

def embed_word_array(vocab, embedding_matrix, word_array, ):
    '''
    Uses the average of the word embeddings of all 
    words in the given word array. 
    ***Assumes that every word it encounters is in the vocabulary!***
    '''
    dim = embedding_matrix.shape[1]
    vector = np.zeros(dim)
    
    # If no words are in the word array, return the zero vector
    if len(word_array) == 0: 
        return vector
    
    # Otherwise embed each word
    for word in word_array:
        word_index = vocab[word]
        vector += embedding_matrix[word_index]
        
    # Compute the average of all embedded words in sentence
    return vector / len(word_array)

def _read_txt_file(file_path, vocab=False, split=False):
    token_array = []
    with open(file_path) as f:
        for line in f:
            if split:
                # If split (test_data) then exclude the number and first comma
                line = " ".join(line.split(",")[1:])
            tokens = [t for t in line.strip().split()]
            # If vocab is given, then check filter out tokens not in vocab
            if vocab:
                tokens = [t for t in tokens if t in vocab] 
            token_array.append(tokens)
    return token_array

def make_text_dataset(vocab, use_full_dataset=False):
    pos_path = constants.POS_PATH
    neg_path = constants.NEG_PATH

    if use_full_dataset == True:
        pos_path = constants.POS_PATH_FULL
        neg_path = constants.NEG_PATH_FULL
    
    print("Reading positive tweets...")
    train_pos = _read_txt_file(pos_path, vocab)

    print("Reading negative tweets...")
    train_neg = _read_txt_file(neg_path, vocab)

    print("Reading test data tweets...")
    test_tweets = _read_txt_file(constants.TEST_DATA_PATH, vocab=vocab, split=True)
    
    # Join negative and positive tweets into one list of word arrays
    train_tweets = train_neg + train_pos 

    # Create the class vector for the train tweets (1 = positive, 0 = negative - works better for learning)
    train_labels = np.array(([0] * len(train_neg)) + ([1] * len(train_pos)))

    dataset = { "train_tweets" : train_tweets, 
                "train_labels" : train_labels, 
                "test_tweets" : test_tweets } 

    return dataset