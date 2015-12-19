#!/bin/bash
set -e
time ./train.py ../trainmodel_set.tsv
time ./predict.py ../localval_set.tsv
echo
./eval.py ../localval_set.tsv prediction.csv
