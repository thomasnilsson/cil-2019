import pickle
import numpy as np 
from word_embedding import WordEmbedding
import constants
import create_baseline_data
import submit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

def train_logistic_regression(batched_train_dataset, embedding, n_estimators=200):
    # Set up Logistic Regression model with Stochastic Gradient Descent
    model = SGDClassifier(loss="log")
    num_batches = len(batched_train_dataset)
    print("Training model...")
    for x_batch, y_batch in tqdm(batched_train_dataset, total=num_batches):
        # Embed training batch
        x_batch_embedded = np.array([
            create_baseline_data.embed_word_array(
                embedding.vocab, embedding.vectors, word_array) 
            for word_array in x_batch
        ])
        model.partial_fit(x_batch_embedded, y_batch, classes=[0,1])
    
    return model  

def validate_model(m, embedding, x_val, y_val):
    print("Embedding validation set...")
    x_val_embedded = np.array([
        create_baseline_data.embed_word_array(embedding.vocab, embedding.vectors, word_array) 
        for word_array in tqdm(x_val)
    ])
    print("Predicting on validation set...")
    y_pred = m.predict(x_val_embedded)
    acc = (y_pred == y_val).mean()
    print("Val accuracy:", acc)

def predict_test(m, embedding, x_test, model_name):
    # Create a new submission file in the 'submission' dir
    print("Embedding test set...")
    # Embed test dataset
    x_test_embedded = np.array([
        create_baseline_data.embed_word_array(embedding.vocab, embedding.vectors, word_array) 
        for word_array in tqdm(x_test)
    ])
    print("Creating submission...")
    y_test = m.predict(x_test_embedded)
    submit.generate_submission_file(y_test, model_name)

def load_and_eval_model(args):
    complete_path = constants.SAVED_BASELINE_MODELS_DIR + args.model_file
    print("Loading model at %s..." % complete_path)
    model = pickle.load(open(complete_path, "rb"))
    embedding, _, _, x_val, y_val, _ = get_baseline_data(args)
    validate_model(model, embedding, x_val, y_val)

def get_baseline_data(args):
    # Load embedding
    embedding = create_baseline_data.load_glove_embedding(args.embedding_type)

    # Load text dataset
    if args.use_full_dataset:
        data_index = pickle.load(open(constants.DATA_INDEX_FULL_PATH, "rb"))
    else:
        data_index = pickle.load(open(constants.DATA_INDEX_SMALL_PATH, "rb"))

    text_dataset = create_baseline_data.make_text_dataset(
        embedding.vocab,
        use_full_dataset=args.use_full_dataset)

    # Split dataset into train, val, and test
    x_train = [text_dataset["train_tweets"][i] for i in data_index["train_index"]]
    y_train = [text_dataset["train_labels"][i] for i in data_index["train_index"]]
    x_val = [text_dataset["train_tweets"][i] for i in data_index["test_index"]]
    y_val = [text_dataset["train_labels"][i] for i in data_index["test_index"]]
    x_test = text_dataset["test_tweets"]

    return embedding, x_train, y_train, x_val, y_val, x_test

def train_model(args):    
    embedding, x_train, y_train, x_val, y_val, x_test = get_baseline_data(args)

    # Split train data into batches to fit into memory
    batch_size = 10000
    split_size = int(len(x_train) / batch_size)
    x_train_batches = np.array_split(x_train, split_size)
    y_train_batches = np.array_split(y_train, split_size)
    batched_train_dataset = list(zip(x_train_batches, y_train_batches))
    
    # Train model in batches
    m = train_logistic_regression(batched_train_dataset, embedding)

    # Create the model name for saving submission and saving the model
    model_name = args.model_type + "_" + args.embedding_type

    # Predict on the validation set
    validate_model(m, embedding, x_val, y_val)

    # Predict on the test set and save submission
    predict_test(m, embedding, x_test, model_name)

    # Save the model
    print("Saving model...")
    path = constants.SAVED_BASELINE_MODELS_DIR + model_name + ".pickle"
    pickle.dump(m, open(path, "wb"))