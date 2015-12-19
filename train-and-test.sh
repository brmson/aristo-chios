#!/bin/sh
set -e
time ./train.py ../trainmodel_set.tsv
time ./predict.py ../localval_set.tsv ; ./eval.py ../localval_set.tsv prediction.csv
