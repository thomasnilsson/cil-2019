import numpy as np

class WordEmbedding(object):
    """
   Generic Word Embedding class to mirror gensim word2vec class
    """

    def __init__(self, vectors, vocab):
        if vectors.shape[0] != len(vocab):
            print("Vocab not same size as embedding matrix!")
        self.vectors = vectors
        self.vocab = vocab
        self.dim = vectors.shape[1]
        print("WordEmbedding initialized with dim=%i and vocab_size=%i" % (self.dim, len(self.vocab)))
        
    def word_vector(self, word):
        try:
            word_index = self.vocab[word]
            v = self.vectors[word_index]
        except KeyError:
            print("Word '%s' was not found in vocabulary, a normal dsitributed vector will be used instead.")
            v = np.random.uniform(low=-0.25, high=0.25, size=self.dim)
        return v
    
    def vocab(self):
        return self.vocab
    
    def vectors(self):
        return self.vectors