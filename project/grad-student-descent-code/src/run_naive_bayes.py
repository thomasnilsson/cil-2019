# Custom modules
from naive_bayes import nb_file_handler
from naive_bayes import nb_model_bag_of_ngrams as model 
from naive_bayes import nb_statistics
from naive_bayes import nb_file_generator
import constants

# Python modules
import json
import os
import numpy as np
from time import perf_counter, process_time


def predict_validation(theta, ngram):
  # Predict on validation set
  print("Predicting on validation set...")
  val_test_data, val_sentiment_data = nb_file_handler.read_val_file(constants.NAIVE_BAYES_VAL_DATA)
  val_labels = np.array([x[1] for x in val_sentiment_data])

  pred_val_data = model.test_naive_bayes(val_test_data, theta, ngram=ngram)
  pred_val = np.array([x[1] for x in pred_val_data])

  print("Validation accuracy:", (val_labels == pred_val).mean())


def predict_test(theta, ngram):
  # Testing
  print("Predicting on validation set...")
  test_data = nb_file_handler.read_test_file(constants.TEST_DATA_PATH)
  predictions = model.test_naive_bayes(test_data, theta, ngram=ngram)

  # Write predictions to file
  print("Saving predicitions to file...")
  nb_file_handler.write_predictions_file(predictions)

def load_and_evaluate_model(ngram=3, smoothing=('add', 0.7)):
  print("Training Naive bayes model (model too large to be serialized)...")
  train(skip_test=True)

def train(ngram=3, smoothing=('add', 0.7), skip_test=False):
  # Example input: ngram=3, smoothing=('add',1)
  # Generate dev and train files (10%/90%)
  data_exists = os.path.isfile(constants.NAIVE_BAYES_TRAIN_DATA)\
                and os.path.isfile(constants.NAIVE_BAYES_VAL_DATA)

  if not data_exists:
    print("Could not find NaiveBayes dataset, generating it...")
    nb_file_generator.generate_val_and_train_files(filename_neg=constants.NEG_PATH_FULL,
                                filename_pos=constants.POS_PATH_FULL,
                                train_filename=constants.NAIVE_BAYES_TRAIN_DATA,
                                val_filename=constants.NAIVE_BAYES_VAL_DATA)

  # Training
  print("Training classifier...")
  train_time = (perf_counter(), process_time())
  train_data = nb_file_handler.read_train_file(constants.NAIVE_BAYES_TRAIN_DATA)
  theta = model.train_naive_bayes(train_data,ngram=ngram, smoothing = smoothing)
  train_time = (perf_counter()-train_time[0], process_time()-train_time[1])

  # Predict on validation dataset
  predict_validation(theta, ngram)

  if not skip_test:
    # predict on test dataset
    predict_test(theta, ngram)
