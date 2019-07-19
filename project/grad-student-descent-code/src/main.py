import argparse
import submit
import rnn_model_manager
import create_rnn_data
import baseline
import constants
import run_naive_bayes
import majority_model

import keras
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

def train(args):
    if args.use_full_dataset:
        print("Using full dataset...")
    else:
        print("Using small dataset...")
    if args.model_type  == "log_reg":
        baseline.train_model(args)
    elif args.model_type[:3] == "rnn":
        rnn_model_manager.train_rnn(args)
    elif args.model_type == "majority":
        majority_model.run(args)
    elif args.model_type == "naive_bayes":
        run_naive_bayes.train()
    else:
        print("Model not recognized!")
        exit(0)

def evaluate(args):
    if args.model_file[:3] == "rnn":
        rnn_model_manager.load_and_evaluate_rnn(args)
    elif args.model_file[:7] == "log_reg":
        baseline.load_and_eval_model(args)
    elif args.model_file == "naive_bayes":
        run_naive_bayes.load_and_evaluate_model()


def preprocess(args):
    create_rnn_data.make_dataset(args.use_full_dataset)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("--action", type=str, default="train", help="What to do, i.e. train or preprocess.")
    parser.add_argument("--env", type=str, default="local", help="Whether to run on GPU or not")
    parser.add_argument("--model_type", type=str, default="dummy", help="model type to evaluate")
    parser.add_argument("--use_full_dataset", type=int, default=0, help="Use full dataset (0: no, 1: yes)")
    parser.add_argument("--embedding_type", type=str, default=constants.GLOVE_STANFORD, 
                        help="Embedding type (stanford or cil)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")
    parser.add_argument("--cell_size", type=int, default=256, help="RNN cell size")
    parser.add_argument("--model_file", type=str, default=None, help="Filename of model (for evaluation)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    
    args = parser.parse_args()

    if args.env == "gpu":
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        keras.backend.set_session(sess)

    print("Args", args)
    
    print("Action chosen:", args.action)

    if args.action == "train":
        train(args)
    elif args.action == "evaluate":
        evaluate(args)
    elif args.action == "preprocess":
        preprocess(args)
    

main()