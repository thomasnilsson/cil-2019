import numpy as np
from collections import Counter
from tqdm import tqdm

def train_data_to_ngrams(train_data, ngram=None):
  assert ngram and ngram > 0, "Specify ngram"
  for i, (sentiment, tweet) in enumerate(train_data):
    for ia in range(ngram-1):
      tweet.insert(0,'__BEGIN__')
      tweet.append('__END__')
    for iw in range(len(tweet)-ngram+1):
      if iw < len(tweet)-ngram+1:
        tweet[iw] = ' '.join(tweet[iw:iw+ngram])
    for ip in range(ngram-1):
      tweet.pop()
    train_data[i] = (sentiment,tweet)


def test_data_to_ngrams(test_data,ngram=None):
  assert ngram and ngram > 0, "Specify ngram"
  for i, (ID,tweet) in enumerate(test_data):
    for ia in range(ngram-1):
      tweet.insert(0,'__BEGIN__')
      tweet.append('__END__')
    for iw in range(len(tweet)-ngram+1):
      if iw < len(tweet)-ngram+1:
        tweet[iw] = ' '.join(tweet[iw:iw+ngram])
    for ip in range(ngram-1):
      tweet.pop()
    test_data[i] = (ID,tweet)

def train_naive_bayes(train_data, ngram=None,smoothing = None):
  assert ngram and ngram > 0, "Specify ngram"
  if ngram > 1:
    train_data_to_ngrams(train_data,ngram)
  # create big neg and pos documents
  # and count N_neg and N_pos
  big_neg = []
  big_pos = []
  N_tot = len(train_data)
  N_neg = 0
  N_pos = 0
  print("Concatenating tweets...")
  for sentiment, tweet in tqdm(train_data):
    if sentiment:
      N_pos += 1
      big_pos.extend(tweet)
    else:
      N_neg += 1
      big_neg.extend(tweet)
  big_all = big_neg + big_pos

  ## Training ##
  # Count prior probabilities estimates
  log_P_prior = (np.log(N_neg/N_tot),np.log(N_pos/N_tot))
  
  # Count maximum likelihood estimate for each word
  print("Counting occurences...")
  C_all = Counter(big_all)
  C_neg = Counter(big_neg)
  C_pos = Counter(big_pos)
  Vocabulary = set(C_all.keys())
  n_Vocabulary = len(C_all)
  loglikelihood = dict()
  
  if smoothing and smoothing[0] == 'add':
    alpha = smoothing[1]
  else:
    alpha = 0
  
  print("Calculating log-likelihoods...")
  for word in tqdm(C_all):
    # neg sentiment
    loglikelihood[(0,word)] = np.log( (C_neg[word]+alpha) / (len(big_neg)+alpha*n_Vocabulary) )
    # pos sentiment
    loglikelihood[(1,word)] = np.log( (C_pos[word]+alpha) / (len(big_pos)+alpha*n_Vocabulary) )
  return log_P_prior, loglikelihood, Vocabulary

def test_naive_bayes(test_data, theta, ngram=None):
  assert ngram and ngram > 0, "Specify ngram"
  log_P_prior, loglikelihood, V = theta
  if ngram > 1:
    test_data_to_ngrams(test_data,ngram=ngram)
  predictions = []
  for ID,tweet in test_data:
    S = np.array(log_P_prior)
    for w in tweet:
      if w in V:
        S[0] += loglikelihood[(0,w)]
        S[1] += loglikelihood[(1,w)]
    pred = np.argmax(S)
    predictions.append((ID,pred))
  return predictions
