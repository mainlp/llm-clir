#!/bin/bash
set -euo pipefail

# LLM-CLIR/reranking/scripts
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# LLM-CLIR/.env
ENV_FILE="${SCRIPT_DIR}/../../.env"

# LLM-CLIR/reranking/pairwise.py
PYTHON_SCRIPT="${SCRIPT_DIR}/../pairwise.py"

set -a
source "$ENV_FILE"
set +a

GPU_ID=1
MODEL="Llama-3.1-8B-Instruct"
DATASET="clef"
RETRIEVER=""

OUTPUT_DIR=""
PASSAGE_LENGTH=300
HITS=100
QUERY_LENGTH=128
SHUFFLE_METHOD=random
SCORING_METHOD=generation
DEVICE=cuda
METHOD=bubblesort
BATCH_SIZE=2
K=10

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gpu) GPU_ID="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;   # clef or ciral
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    --passage_length) PASSAGE_LENGTH="$2"; shift ;;
    --retriever) RETRIEVER="$2"; shift ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

OUTPUT_DIR="${OUTPUT_DIR:-${RERANKING_BASE}/rankings/pairwise/${DATASET}}"
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=$GPU_ID

if [[ ${MODEL,,} =~ ^llama ]]; then
  MODEL_PATH="meta-llama/${MODEL}"
  TOKENIZER_PATH="meta-llama/${MODEL}"
else
  MODEL_PATH="CohereLabs/${MODEL}"
  TOKENIZER_PATH="CohereLabs/${MODEL}"
fi

echo $MODEL_PATH


if [[ "$DATASET" == "ciral" ]]; then
  langs=(hausa somali swahili yoruba)
else
  langs=(german_finnish german_italian german_russian english_german english_finnish english_italian english_russian finnish_italian finnish_russian)
fi

to_model_token() {
  local m="$1"
  shopt -s nocasematch
  if [[ "$m" == llama-3.1* || "$m" == Llama-3.1* ]]; then
    echo "llama_3_1"
  elif [[ "$m" == aya-101* || "$m" == aya101* ]]; then
    echo "aya-101"
  fi
  shopt -u nocasematch
}

model_token="$(to_model_token "$MODEL")"

for lang in "${langs[@]}"; do
  qlang=$(echo $lang | cut -d '_' -f 1)
  dlang=$(echo $lang | cut -d '_' -f 2 | cut -d '.' -f 1)
  
  query_file="${DATA_LOCATION}/${DATASET}/queries/${qlang}.jsonl"
  
  if [[ ${DATASET} == "ciral" ]]; then
    candidates_file="${RERANKING_BASE}/processed_pairwise/${DATASET}/${RETRIEVER}/${dlang}.tsv"
  else
    candidates_file="${RERANKING_BASE}/processed_pairwise/${DATASET}/${RETRIEVER}/${qlang}_${dlang}.tsv"
  fi
  
  cand_parent="$(basename "$(dirname "$candidates_file")")"   # e.g., BM25_OG
  retriever="${cand_parent%%_*}"
  suffix="${cand_parent#*_}"                          

  lang_dir="${OUTPUT_DIR}/${retriever}/${model_token}_${suffix}"
  save_path="${lang_dir}/${lang}.trec"
  mkdir -p "$lang_dir"

  echo "==============================="
  echo "Running: save to $save_path"
  echo "Query file: $query_file"
  echo "Candidates file: $candidates_file"
  echo "Save path: $save_path"
  echo "==============================="

  python3 "$PYTHON_SCRIPT" \
    run --model_name_or_path "$MODEL_PATH" \
        --tokenizer_name_or_path "$TOKENIZER_PATH" \
        --save_path "$save_path" \
        --query_file "$query_file" \
        --candidates_file "$candidates_file" \
        --hits $HITS \
        --query_length $QUERY_LENGTH \
        --passage_length $PASSAGE_LENGTH \
        --shuffle_ranking $SHUFFLE_METHOD \
        --scoring $SCORING_METHOD \
        --device $DEVICE \
        --dataset $DATASET \
    pairwise --method $METHOD \
              --batch_size $BATCH_SIZE \
              --k $K
done
