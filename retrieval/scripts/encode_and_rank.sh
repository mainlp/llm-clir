#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

set -a
source $SCRIPT_DIR/../../.env
set +a

DEVICE=0
ENCODER="nv"
DATASET="clef"

while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --encoder)
      ENCODER="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--device GPU_ID] [--dataset DATASET_NAME] [--encoder ENCODER_TYPE]"
      echo "  GPU_ID: ID of the GPU to use (default: 0)"
      echo "  DATASET_NAME: Dataset to use for the evaluation"
      echo "  ENCODER_TYPE: Encoding model to use (choices: mgte, m3, e5, nv) (default: nv)"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# note: LANGUAGE_PAIRS for ciral look monolingual, but query files contain english queries
if [[ "$DATASET" == "ciral" ]]; then
    LANGUAGES=(
        "somali"
        "swahili"
        "yoruba"
        "hausa"
    )
    LANGUAGE_PAIRS=(
        "SO-SO"
        "SW-SW"
        "YO-YO"
        "HA-HA"
    )
elif [[ "$DATASET" == "clef" ]]; then
    LANGUAGES=(
        "english"
        "finnish"
        "german"
        "italian"
        "russian"
    )
    LANGUAGE_PAIRS=(
        "EN-FI"
        "EN-IT"
        "EN-RU"
        "EN-DE"
        "DE-FI"
        "DE-IT"
        "DE-RU"
        "FI-IT"
        "FI-RU"
    )
else
    echo "Error: Unsupported dataset '$DATASET'. Please use 'ciral' or 'clef'."
    exit 1
fi

# Map language codes to full names for file paths
declare -A LANG_MAP=(
  ["EN"]="english"
  ["DE"]="german"
  ["FI"]="finnish"
  ["IT"]="italian"
  ["RU"]="russian"
  ["SO"]="somali"
  ["SW"]="swahili"
  ["YO"]="yoruba"
  ["HA"]="hausa"
)

# create dir for results
mkdir -p "results/${DATASET}/${ENCODER}/"

echo 'Encode docs'
for lang in "${LANGUAGES[@]}"; do
CUDA_VISIBLE_DEVICES=$DEVICE python encode.py \
    --lang ${lang} \
    --dataset ${DATASET} \
    --encoder ${ENCODER}
done

echo 'Encode queries'
for lang in "${LANGUAGES[@]}"; do
CUDA_VISIBLE_DEVICES=$DEVICE python encode.py \
    --lang ${lang} \
    --encoder ${ENCODER} \
    --dataset ${DATASET} \
    --encode_queries
done

for pair in "${LANGUAGE_PAIRS[@]}"; do
    SRC=${pair%-*}
    TGT=${pair#*-}

    echo "Ranking $pair"

    src_full=${LANG_MAP[$SRC]}
    tgt_full=${LANG_MAP[$TGT]}

    # Run the retriever
    python rank.py \
        --qry_lang  ${src_full} \
        --doc_lang  ${tgt_full} \
        --encoder ${ENCODER} \
        --dataset ${DATASET} \

    if [[ ${DATASET} == "ciral" ]]; then
      TREC_FILE="${LANG_MAP[$SRC]}.trec"
      MEASURE=nDCG@20
    else
      TREC_FILE="${SRC}_${TGT}.trec"
      MEASURE=MAP
    fi

    # Evaluate
    echo "Evaluating $TREC_FILE"

    ir_measures ${DATA_LOCATION}/${DATASET}/qrels/${tgt_full}.txt \
      ${RETRIEVAL_BASE}/rankings/${DATASET}/${ENCODER}/${TREC_FILE} \
      $MEASURE | tee -a ${RETRIEVAL_BASE}/results/${DATASET}/${ENCODER}/${TREC_FILE}.txt
done
