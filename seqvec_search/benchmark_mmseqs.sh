#!/bin/bash

# Benchmark MMseqs2, recording wall time and logs

# Setup
mkdir -p data/mmseqs_dbs data/mmseqs_results data/mmseqs_align data/mmseqs_prefilter
mmseqs createdb data/pfam-dist/train.fasta data/mmseqs_dbs/train
mmseqs createdb data/pfam-dist/test.fasta data/mmseqs_dbs/test

# Time is measured as wall time and written to a file as seconds
# https://unix.stackexchange.com/a/52347/77322

# Run mmseqs at different sensitivities as baseline. Record stdout/stderr and wall time
for i in {1,2,3,4,5,6,7,7.5,8}; do
    echo "Processing sensitivity $i"
    start=$(date +%s)
    mmseqs search -e 10000 -s $i data/mmseqs_dbs/test data/mmseqs_dbs/train data/mmseqs_results/s${i} /tmp &>data/mmseqs_results/s${i}-log.txt
    end=$(date +%s)
    echo $((end - start)) >data/mmseqs_results/s${i}-time.txt
done

# Measure prefilter and align independently
for i in {1,2,3,4,5,6,7,7.5,8}; do
    echo "Processing sensitivity $i"
    mkdir -p data/mmseqs_dbs/s${i}
    start=$(date +%s)
    mmseqs prefilter data/mmseqs_dbs/test data/mmseqs_dbs/train data/mmseqs_prefilter/s${i} -s $i &>data/mmseqs_prefilter/s${i}-log.txt
    end=$(date +%s)
    echo $((end - start)) >data/mmseqs_prefilter/s${i}-time.txt
    start=$(date +%s)
    mmseqs align data/mmseqs_dbs/test data/mmseqs_dbs/train data/mmseqs_prefilter/s${i} data/mmseqs_align/s${i} -e 10000 &>data/mmseqs_align/s${i}-log.txt
    end=$(date +%s)
    echo $((end - start)) >data/mmseqs_align/s${i}-time.txt
done
