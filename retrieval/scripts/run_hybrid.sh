#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RANKINGS_DIR="${SCRIPT_DIR}/../rankings"
DATASET=$1
DENSE_MODEL=$2

OUT_DIR=$RANKINGS_DIR/$DATASET/hybrid

if [[ "$DATASET" == "ciral" ]]; then
  langs=(hausa somali swahili yoruba)
else
  langs=(german_finnish german_italian german_russian english_german english_finnish english_italian english_russian finnish_italian finnish_russian)
fi

mkdir -p $OUT_DIR

for LANG in "${langs[@]}"; do
  echo "Rank averaging bm25 and $DENSE_MODEL ($LANG)"
  python ../hybrid.py --bm25 $RANKINGS_DIR/$DATASET/bm25/$LANG.trec --dense $RANKINGS_DIR/$DATASET/$DENSE_MODEL/$LANG.trec --output $OUT_DIR/$LANG.trec
done

echo "Done"
