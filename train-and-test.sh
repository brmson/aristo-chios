#!/bin/bash
set -e
if false; then
	# only when tokenization or embedding changes
	./train_glove.py ../trainmodel_set.tsv data/glovetrain
	(here=$(pwd)
		cd ~/brmson/Sentence-selection
		./std_run.sh -p "$here"/data/glovetrain
		cp data/Mbtemp.txt "$here"/data/gloveMb.txt
	)
fi
time ./train.py ../trainmodel_set.tsv
echo
echo - Training set:
time ./predict.py ../trainmodel_set.tsv
./eval.py ../trainmodel_set.tsv prediction.csv
echo
echo - Local validation set:
time ./predict.py ../localval_set.tsv
./eval.py ../localval_set.tsv prediction.csv
