#!/bin/sh
# Capture script to run and job name
SCRIPT=$1
JOB_NAME=${2:-unnamed}

# Set up results dir and output file with timestamp
result_dir="result/"
mkdir -p $result_dir
timestamp=$(date +%s)
result_file=$PWD/result/$JOB_NAME\_$timestamp.out

# Submit job
bsub -J $JOB_NAME -oo $result_file -n 6 -N -W 8:00 -R "rusage[mem=2048, ngpus_excl_p=1]" < $SCRIPT

# Confused?
## Link to all BSUB options
## https://www.ibm.com/support/knowledgecenter/en/SSETD4_9.1.3/lsf_command_ref/bsub.heading_options.1.html
