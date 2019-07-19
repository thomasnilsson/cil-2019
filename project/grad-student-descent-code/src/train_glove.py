from scipy.sparse import *
import numpy as np
import pickle
import random
import constants
from tqdm import tqdm
import argparse
from word_embedding import WordEmbedding

def run(dim, epochs):
    cooc = _load_cooc_matrix()
    vocab = _load_vocab()
    print(vocab)
    word_vectors, _ = _train_embeddings(cooc, dim=dim, epochs=epochs)
    word_emb = WordEmbedding(word_vectors, vocab)

    # Save results to file
    path_word_emb = constants.GLOVE_EMBEDDING_CIL_PATH
    pickle.dump(word_emb, open(path_word_emb, "wb"))
    print("Finished saving embeddings at %s" % path_word_emb)

def _load_vocab(path="../twitter_data/"):
    tokens = []
    with open(path + "vocab_cut.txt") as f:
        for line in f:
            tokens.append(line)
    return {word: idx for idx,word in enumerate(tokens)}

def _load_cooc_matrix(path="../twitter_data/"):
    print("loading cooccurrence matrix")
    with open(path + 'cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))
    return cooc

def _train_embeddings(cooc, dim=20, n_max=100, epochs=10, learning_rate=0.001):
    print("GloVe with dim=%i, nmax=%i epochs=%i learning_rate=%f" % (dim, n_max, epochs, learning_rate))
    x = np.random.normal(size=(cooc.shape[0], dim))
    y = np.random.normal(size=(cooc.shape[1], dim))

    # Alpha: dampening factor for frequent occuring words
    alpha = 3/4 
    
    # Zip indices with the contents of the cooc_matrix
    data = enumerate(zip(cooc.row, cooc.col, cooc.data))

    print("Training embeddings...")
    for epoch in range(epochs):
        print("Epoch %i/%i" % (epoch+1, epochs))
        for idx, (i, j, n_ij) in tqdm(data, total=cooc.nnz):
            # Weight of word f(n_ij)
            f_ij = min(1.0, (n_ij / n_max)**alpha)
            
            # Compute target and prediction
            # Clip inner product to avoid exploding gradient
            y_target = np.log(n_ij)
            y_pred = np.dot(x[i], y[j]).clip(-10, 10)
            
            # Compute shared gradient
            loss = 2 * learning_rate * f_ij * (y_target - y_pred)
            
            # Update parameters
            x[i] += loss * x[i]
            y[j] += loss * y[j]

    print("Finishied training!")
    return x, y

parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument("--dim", type=int, default=200, help="Embedding dim")
parser.add_argument("--epochs", type=int, default=10, help="Epochs for training")
args = parser.parse_args()

run(args.dim, args.epochs)