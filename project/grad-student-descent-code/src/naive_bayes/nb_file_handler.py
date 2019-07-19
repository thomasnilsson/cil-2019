import re
from datetime import datetime
import constants

def read_train_file(filename):
  train_data = []
  with open(filename) as f:
    for line in f:
      m = re.match(r'\[([0-1])\](.*)',line)
      sentiment = int(m.group(1))
      tweet_str = m.group(2)
      tweet = tweet_str.split()
      train_data.append((sentiment,tweet))
  return train_data

def read_val_file(filename):
  val_test_data = []
  val_sentiment_data = []
  with open(filename,'r') as f:
    for ID,line in enumerate(f):
      m = re.match(r'\[([0-1])\](.*)',line)
      sentiment = int(m.group(1))
      tweet_str = m.group(2)
      tweet = tweet_str.split()
      val_test_data.append((ID,tweet))
      val_sentiment_data.append((ID,sentiment))
  return val_test_data, val_sentiment_data

def read_test_file(filename):
  test_data = []
  with open(filename) as f:
    for line in f:
      m = re.match(r'(\d+),(.*)',line)
      ID = int(m.group(1))
      tweet_str = m.group(2)
      tweet = tweet_str.split()
      test_data.append((ID,tweet))
  return test_data

def write_predictions_file(predictions):
  timestamp = datetime.now().strftime('%B%d_%H%M')
  filename = constants.SUBMISSION_DIR + f'nb_predictions[{timestamp}].txt'

  with open(filename,'w') as f:
    f.write('Id,Prediction\n')
    for i,pred in predictions:
      p = 1 if pred else -1
      f.write(f'{i},{p}\n')
