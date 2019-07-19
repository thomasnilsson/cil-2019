import numpy as np
import random
import pickle
import constants
from tqdm import tqdm

def generate_val_and_train_files(filename_neg=None,
                                 filename_pos=None,
                                 train_filename=None,
                                 val_filename=None):
  all_data = []
  print("Reading raw text files...")
  with open(filename_neg,'r') as f:
    for line in tqdm(f):
      all_data.append((0,line))
  with open(filename_pos,'r') as f:
    for line in tqdm(f):
      all_data.append((1,line))
  
  print("Splitting data...") 
  data_index = pickle.load(open(constants.DATA_INDEX_FULL_PATH, "rb"))
  train_index = data_index["train_index"]
  val_index = data_index["test_index"]
  train_data = [all_data[i] for i in tqdm(train_index)]
  val_data = [all_data[i] for i in tqdm(val_index)]
  
  print("Writing test data to file...")  
  with open(train_filename,'w') as f:
    for sentiment,tweet in tqdm(train_data):
      f.write(f'[{sentiment}]{tweet}')
  
  print("Writing val data to file...")  
  with open(val_filename,'w') as f:
    for sentiment, tweet in tqdm(val_data):
      f.write(f'[{sentiment}]{tweet}')
