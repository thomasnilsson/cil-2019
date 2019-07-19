from word_embedding import WordEmbedding
import constants
import matplotlib.pyplot as plt
import numpy as np
import pickle
from nltk.probability import FreqDist
import os
from tqdm import tqdm

# Tokens for parsing sentences
UNK = "<unk>"
PAD = "<pad>"

def _read_txt_file(file_path, split=False):
    '''Reads all tokens in a file, returns a string matrix. Does not remove duplicate lines'''
    token_array = []
    with open(file_path) as f:
        for line in f:
            if split:
                # If split (test_data) then exclude the number and first comma
                line = " ".join(line.split(",")[1:])
            token_array.append([t for t in line.strip().split()])
    return token_array

def _make_text_dataset(use_full_dataset=False):
    if use_full_dataset:
        neg_path = constants.NEG_PATH_FULL 
        pos_path = constants.POS_PATH_FULL
        print("Loading full dataset...")
    else:
        neg_path = constants.NEG_PATH
        pos_path = constants.POS_PATH
        print("Loading partial dataset...")
    
    # Load txt
    neg = _read_txt_file(neg_path)
    pos = _read_txt_file(pos_path)
    train_tweets = neg + pos
    train_labels = np.array(([0]*len(neg)) + ([1]*len(pos)))
    
    test_tweets = _read_txt_file(constants.TEST_DATA_PATH, split=True)
    
    # Save dataset as dict
    dataset = {"train_tweets" : train_tweets, 
                "train_labels" : train_labels,
                "test_tweets" : test_tweets}
    
    return dataset

# Taken from the NLU course
def _make_embeddings(word_embedding, vocab, verbose=False):
    '''
    Make an embedding matrix from a pretained embedding (WordEmbedding class)
    as well as a vocabulary. Unknown words are given a random
    normal distributed vector as embedding (this will be trained later on).
    '''
    vocab_size = len(vocab)
    matches = 0
    external_embedding = np.zeros(shape=(vocab_size, word_embedding.dim), dtype=np.float32)

    for tok, idx in vocab.items():
        # If words exist in pretrained embedding
        if tok in word_embedding.vocab:
            external_embedding[idx] = word_embedding.word_vector(tok)
            matches += 1
                
    if verbose:
        print("%.2f percent of the words in the vocabulary \
            were found in the embedding" % (100 * matches/vocab_size))
    
    return external_embedding

def _make_capped_word_index(stanford, dataset, vocab_size=20000):
    len_stanford = len(stanford.vocab)
    print("Stanford vocab original length:", len_stanford)
    print("Capped vocab (20k) fraction:", vocab_size / len_stanford)

    # Combine all tweets into one big array of words
    flatten = lambda l: [item for sublist in l for item in sublist]
    dataset_corpus = flatten(dataset["train_tweets"])

    # Count most frequent words 
    fd = FreqDist(dataset_corpus)
    top20k = fd.most_common(vocab_size)
    
    # Make new word index (vocab) with PAD and UNK as the first two tokens (ordering is important)
    top20k_words = [PAD, UNK]
    top20k_words += [x[0] for x in top20k]
    word_index_20k = {word: idx for idx,word in enumerate(top20k_words)}

    return word_index_20k


def unk_sub_array(word_array, vocab):
    '''Replaces all unknown words in a word array with <unk>.'''
    return [word if word in vocab else UNK for word in word_array]

def word_to_id_array(word_array, vocab):
    '''Replaces all words in a word array with their index in the vocab.'''
    return [vocab[word] for word in word_array]

def pad_sequence(id_sentence, word_index, max_length=40):
    '''Pads a sequence, post style'''
    # Number of words to overwrite (max 40)
    n = min(len(id_sentence), max_length)
    
    try:
        # Clip sentence to max words
        clipped_sentence = id_sentence[:n]

        # Init <PAD> vector
        pad_id = word_index[PAD] # Pad id should be 0, but doesnt matter
        padded_seq = np.ones(max_length, dtype=np.int) * pad_id # Should result in a zero array

        # Overwrite the last n elements in the <PAD> vector
        padded_seq[-n:] = clipped_sentence

        return padded_seq
    except:
        print(id_sentence)

def preprocess_dataset(tweets, vocab):
    """
    1. Replace unknown words
    2. Replace words with ids
    3. Pads tweets to length 40
    """
    tweets_with_unks = [unk_sub_array(x, vocab) for x in tqdm(tweets)]
    id_tweets = [word_to_id_array(x, vocab) for x in tqdm(tweets_with_unks)]
    padded_tweets = np.array([pad_sequence(x, vocab) for x in tqdm(id_tweets)])
    return padded_tweets

def make_id_dataset(dataset, vocab, try_load=True):
    print("Making id dataset...")
    padded_train_tweets = preprocess_dataset(dataset["train_tweets"], vocab)
    padded_test_tweets = preprocess_dataset(dataset["test_tweets"], vocab)
    
    id_dataset = {"train_tweets" : padded_train_tweets, 
                    "train_labels" : dataset["train_labels"],
                    "test_tweets" : padded_test_tweets}
    
    return id_dataset

def make_dataset(use_full_dataset=True):
    # Make txt dataset
    txt_dataset = _make_text_dataset(use_full_dataset=use_full_dataset)
    
    # Load full stanford embedding from file
    stanford = pickle.load(open(constants.GLOVE_EMBEDDING_STANFORD_PATH, "rb"))

    # Create vocabulary, cut at top 20k words
    word_index = _make_capped_word_index(stanford, txt_dataset)

    # Reduce embedding matrix to include top 20k words
    embedding_vectors = _make_embeddings(stanford, word_index)
    
    # Create ID dataset, with <pad>'s and <unk>'s
    id_dataset = make_id_dataset(txt_dataset, word_index)

    print("Creating and saving indices...")
    # Make ordering indices for shuffling data
    N = len(id_dataset["train_tweets"])
    index = np.arange(N)
    np.random.seed(constants.SEED)
    np.random.shuffle(index)

    # Divide indices into train and test indices
    divider = int(constants.SPLIT_RATIO * N)
    train_index = index[:divider]
    test_index = index[divider:]
    data_index = {"train_index" : train_index, "test_index" : test_index}

    # Save data index
    index_path = constants.DATA_INDEX_SMALL_PATH
    if use_full_dataset: index_path = constants.DATA_INDEX_FULL_PATH
    pickle.dump(data_index, open(index_path, "wb"))

    # Save to pickle
    print("Saving word embeddings...")
    word_embedding_20k = WordEmbedding(embedding_vectors, word_index)
    pickle.dump(word_embedding_20k, open(constants.STANFORD_20K_EMBEDDING_PATH, "wb"))
    
    print("Saving txt dataset...")
    txt_dataset_path = constants.TXT_DATASET_SMALL_PATH
    if use_full_dataset: txt_dataset_path = constants.TXT_DATASET_FULL_PATH
    pickle.dump(txt_dataset, open(txt_dataset_path, "wb"))

    print("Saving id dataset...")
    id_dataset_path = constants.ID_DATASET_SMALL_PATH
    if use_full_dataset: id_dataset_path = constants.ID_DATASET_FULL_PATH
    pickle.dump(id_dataset, open(id_dataset_path, "wb"))

    # Plot tweet length distribution
    tweets_lengths = np.array([len(tweet) for tweet in txt_dataset["train_tweets"]])
    plt.hist(tweets_lengths, bins=50, edgecolor="black")
    plt.xlabel("Tweet length")
    plt.ylabel("Frequency")
    plt.savefig(constants.PLOTS_DIR + "tweet_lengths.eps", format="eps", dpi=1000, bbox_inches="tight")
    
    # Print tweet fraction with length <= 40 words
    frac_max_40_words = len(tweets_lengths[tweets_lengths <= 40]) / len(tweets_lengths)
    print("Fraction of tweets with length <= 40 words:", frac_max_40_words)

