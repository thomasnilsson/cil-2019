#!/bin/bash
function download_files() {
	# Downloads files to current directory
	url=$1
	files=( "$@" )

	# Check if wget is installed
	if [ -x "$(command -v wget)" ]; then
		echo 'Using wget.\n'
		for f in "${files[@]}";
		do 
			echo "Downloading " $f "...\n"
			wget -O $f "$url$f"
		done
	# Check if cURL is installed
	elif [ -x "$(command -v curl)" ]; then
		echo 'Using cURL.\n'
		for f in "${files[@]}";
		do 
			echo "Downloading " $f "...\n"
			curl -o $f "$url$f"
		done
	# Otherwise exit program
	else
		echo 'Error: wget is not installed either. Please download the files manually from https://polybox.ethz.ch/index.php/s/pCbnQIoY4AisA4h'
		exit 1
	fi
}

# Make directories which are used later
mkdir data
mkdir data/full
mkdir twitter_data
mkdir embeddings
mkdir saved_models
mkdir saved_models/baseline_models
mkdir saved_models/rnn_models

##### Download required twitter data files
data_full_url='https://polybox.ethz.ch/index.php/s/pCbnQIoY4AisA4h/download?path=%2Fdata%2Ffull&files='
twitter_data_url='https://polybox.ethz.ch/index.php/s/pCbnQIoY4AisA4h/download?path=%2Ftwitter_data&files='
embedding_url='https://polybox.ethz.ch/index.php/s/pCbnQIoY4AisA4h/download?path=%2Fembeddings&files='
saved_baseline_models_url='https://polybox.ethz.ch/index.php/s/F4LsN1aAGJ6wZbW/download?path=%2Fbaseline_models&files='
saved_rnn_models_url='https://polybox.ethz.ch/index.php/s/F4LsN1aAGJ6wZbW/download?path=%2Frnn_models&files='

twitter_data_files=(	\
	'sample_submission.csv' \
	'test_data.txt' \
	'train_neg.txt' \
	'train_neg_full.txt' \
	'train_pos.txt' \
	'vocab_cut.txt'\
	'train_pos_full.txt'\
	'train_pos_full.txt'\
	'train_pos_full.txt'\
	);

embedding_files=(\
	'glove_cil_200.WordEmbedding'\
	'glove_stanford_200.WordEmbedding.zip'\
	'stanford_20k.WordEmbedding');

data_full_files=(\
	'id_dataset_full.dict'\
	'id_dataset_small.dict'\
	'data_index_full.dict'\
	'data_index_small.dict'\
	'txt_dataset_full.dict'\
	'txt_dataset_small.dict'\
	);

baseline_models_files=(\
	'log_reg_cil.pickle'\
	'log_reg_stanford.pickle'\
	);

rnn_models_files=(\
	'val_prediction_matrix.obj'\
	'rnn_simple_256.hdf5'\
	'rnn_pooling_256.hdf5'\
	'rnn_pooled_attention_512.hdf5'\
	'rnn_pooled_attention_256.hdf5'\
	'rnn_conv_256.hdf5'\
	'rnn_attention_256.hdf5'\
	);

### EMBEDDING FILES
cd embeddings
download_files $embedding_url "${embedding_files[@]}"
unzip glove_stanford_200.WordEmbedding.zip
cd ..

### DATA FILES
cd data/full
download_files $data_full_url "${data_full_files[@]}"
cd ../..

### TWITTER DATA FILES
cd twitter_data
download_files $twitter_data_url "${twitter_data_files[@]}"
cd ..

### BASELINE MODELS
cd saved_models/baseline_models
download_files $saved_baseline_models_url "${baseline_models_files[@]}"
cd ../..

### RNN MODELS
cd saved_models/rnn_models
download_files $saved_rnn_models_url "${rnn_models_files[@]}"
cd ../..


### DONE
echo "-- Finished downloads. --"
