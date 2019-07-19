GLOVE_STANFORD = "stanford"
GLOVE_CIL = "cil"
WORD2VEC = "word2vec"

uniq = "" # Whether or not to use uniq dataset (set to "" for full)
if uniq:
    data_subdir = "uniq/"
else:
    data_subdir = "full/"

DATA_DIR = "../data/" + data_subdir
TWITTER_DATA_DIR = "../twitter_data/"
EMBEDDINGS_DIR = "../embeddings/"
SUBMISSION_DIR = "../submissions/"
SAVED_MODELS_DIR = "../saved_models/"
SAVED_BASELINE_MODELS_DIR = SAVED_MODELS_DIR + "baseline_models/"
SAVED_RNN_MODELS_DIR = SAVED_MODELS_DIR + "rnn_models/"
PLOTS_DIR = "../plots/"


# RAW DATA
VOCAB_PATH = TWITTER_DATA_DIR + "vocab.pkl"
POS_PATH = TWITTER_DATA_DIR + "train_pos" + uniq + ".txt"
NEG_PATH = TWITTER_DATA_DIR + "train_neg" + uniq + ".txt"
POS_PATH_FULL = TWITTER_DATA_DIR + "train_pos_full" + uniq + ".txt"
NEG_PATH_FULL = TWITTER_DATA_DIR + "train_neg_full" + uniq + ".txt"
TEST_DATA_PATH = TWITTER_DATA_DIR + "test_data.txt"
SAMPLE_SUBMISSION_PATH = TWITTER_DATA_DIR + "sample_submission.csv"

# DATASETS
DATASET_W2V_PATH = DATA_DIR + "dataset_w2v.pickle"
DATASET_GLOVE_CIL_PATH = DATA_DIR + "dataset_glove_cil.pickle"
DATASET_GLOVE_STANFORD_PATH = DATA_DIR + "dataset_glove_stanford.pickle"

ID_DATASET_FULL_PATH = DATA_DIR + "id_dataset_full" + uniq + ".dict"
ID_DATASET_SMALL_PATH = DATA_DIR + "id_dataset_small" + uniq + ".dict"

TXT_DATASET_FULL_PATH = DATA_DIR + "txt_dataset_full" + uniq + ".dict"
TXT_DATASET_SMALL_PATH = DATA_DIR + "txt_dataset_small" + uniq + ".dict"

DATA_INDEX_FULL_PATH = DATA_DIR + "data_index_full" + uniq + ".dict"
DATA_INDEX_SMALL_PATH = DATA_DIR + "data_index_small" + uniq + ".dict"

NAIVE_BAYES_TRAIN_DATA = DATA_DIR + "train_data_nb" + uniq + ".txt"
NAIVE_BAYES_VAL_DATA = DATA_DIR + "val_data_nb" + uniq + ".txt"

# EMBEDDINGS
WORD2VEC_EMBEDDING_PATH = EMBEDDINGS_DIR + "GoogleNews-vectors-negative300.bin"
GLOVE_EMBEDDING_STANFORD_PATH = EMBEDDINGS_DIR + "glove_stanford_200.WordEmbedding"
GLOVE_EMBEDDING_CIL_PATH = EMBEDDINGS_DIR + "glove_cil_200.WordEmbedding"
STANFORD_20K_EMBEDDING_PATH = EMBEDDINGS_DIR + "stanford_20k.WordEmbedding"

# Model path
NAIVE_BAYES_LOGPRIOR_PATH = SAVED_BASELINE_MODELS_DIR + "naive_bayes_logprior.json"
NAIVE_BAYES_LOGLIKELIHOOD_PATH = SAVED_BASELINE_MODELS_DIR + "naive_bayes_loglikelihood.json"
NAIVE_BAYES_VOCAB_PATH = SAVED_BASELINE_MODELS_DIR + "naive_bayes_vocab.json"

# Various
SEED = 42
SPLIT_RATIO = 0.9 # 90% for train data