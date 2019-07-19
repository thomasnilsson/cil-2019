# Train baseline models
#sh train_glove.sh
sh log_reg_cil.sh
sh log_reg_stanford.sh
sh naive_bayes.sh # This also evaluates the model

# Train RNN models
sh rnn_simple.sh
sh rnn_pooling.sh
sh rnn_conv.sh
sh rnn_attention.sh
sh rnn_pooled_attention.sh
sh rnn_pooled_attention_512.sh


