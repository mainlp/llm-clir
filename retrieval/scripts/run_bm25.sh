#!/bin/bash

DATASET=$1
QLANG=$2
DLANG=$3

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
RETRIEVAL_DIR=$(dirname "$SCRIPT_DIR")

code2full () {
  case "$1" in
    de) echo "german" ;;
    fi) echo "finnish" ;;
    en) echo "english" ;;
    ru) echo "russian" ;;
    it) echo "italian" ;;
    ha) echo "hausa" ;;
    yo) echo "yoruba" ;;
    sw) echo "swahili" ;;
    so) echo "somali" ;;
    *)  echo "$1" ;;
  esac
}

# We always use DT (QT results are taken from (Adeyemi et al., 2024))
TRANSLATION="DT"
INDEX_LANG=$QLANG

echo "Running $QLANG $DLANG"

echo "Reformatting document files"
cmd="python $RETRIEVAL_DIR/bm25_reformat.py $DATASET $QLANG $DLANG $TRANSLATION"
echo $cmd
eval "$cmd"

INDEX_DIR="$RETRIEVAL_DIR/indexes/$DATASET/${QLANG}2${DLANG}"
echo "Creating Lucene index"
index_cmd="""python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $INDEX_DIR \
  --language $INDEX_LANG \
  --index $INDEX_DIR \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw"""
echo $index_cmd
eval "$index_cmd"

echo "Running bm25 retrieval"
retrieve_cmd="python $RETRIEVAL_DIR/bm25_retrieve.py $DATASET $QLANG $DLANG $TRANSLATION $INDEX_DIR"
echo $retrieve_cmd
eval "$retrieve_cmd"

echo $DATASET
if [[ ${DATASET} == "ciral" ]]; then
  TREC_FILE=$(code2full "$DLANG")".trec"
  MEASURE="nDCG@20"
else
  TREC_FILE=$(code2full ""$QLANG)"_"$(code2full "$DLANG")".trec"
  MEASURE="MAP"
fi

RANKING_DIR="$RETRIEVAL_DIR/rankings/$DATASET/bm25/$TREC_FILE"
QRELS_FILE="$RETRIEVAL_DIR/../data/${DATASET}/qrels/$(code2full "$DLANG").txt"

eval_cmd="ir_measures $QRELS_FILE $RANKING_DIR $MEASURE"
echo $eval_cmd
eval "$eval_cmd"
