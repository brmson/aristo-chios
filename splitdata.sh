#!/bin/sh

read header
echo "$header" >../trainmodel_set.tsv
echo "$header" >../localval_set.tsv
i=0
while read line; do
	if [ $((i%5)) -eq 0 ]; then
		echo "$line" >>../localval_set.tsv
	else
		echo "$line" >>../trainmodel_set.tsv
	fi
	i=$((i+1))
done
