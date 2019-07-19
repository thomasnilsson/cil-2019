from datetime import datetime

def val_data_write_results(val_sentiment_data,predictions,ngram=None,smoothing=None, raport_filename=None, time=None):
  n_tot = 0
  n_pos = 0
  n_neg = 0
  n_correct = 0
  n_correct_pos = 0
  n_correct_neg = 0
  for (ID_d,s), (ID_p, pred) in zip(val_sentiment_data, predictions):
    assert ID_d == ID_p, 'IDs in different order'
    n_tot += 1
    if s == 0:
      n_neg += 1
      if s == pred:
        n_correct += 1
        n_correct_neg += 1
    elif s == 1:
      n_pos += 1
      if s == pred:
        n_correct += 1
        n_correct_pos += 1
    else:
      assert False, 'is something wrong with you'
  
  if raport_filename:
    filename = raport_filename
  else:
    filename = 'raport_val_test_results.txt'

  timestamp = datetime.now().strftime('%B%d_%H%M')
  with open(filename,'a') as f:
    if smoothing:
      s_type, s_factor = smoothing
    else:
      s_type, s_factor = ('add',0)
      
    wrt_str = f'[{timestamp}] ngram: {ngram}, smoothing: ({s_type}-{s_factor})'
    wrt_str += f', n_correct/n_tot: {n_correct}/{n_tot} [{n_correct/n_tot:.4f}]'
    wrt_str += f', n_correct_neg/n_neg: {n_correct_neg}/{n_neg} [{n_correct_neg/n_neg:.4f}]'
    wrt_str += f', n_correct_pos/n_pos: {n_correct_pos}/{n_pos} [{n_correct_pos/n_pos:.4f}]'
    wrt_str += f', n_neg/n_pos: {n_neg}/{n_pos} [{n_neg/n_pos:.4f}]'
    if time:
      wrt_str += f', train_time: [{time[0]:.2f} (s, wall clock), {time[1]:.2f} (s, process time)]'
    f.write(f'{wrt_str}\n')
    print(wrt_str)

