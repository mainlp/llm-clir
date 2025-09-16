#!/bin/bash
# set this to your local path (project root path), this line will be overwritten by setup_repllama.sh
PROJECT_ROOT="/path/to/projects/LLM-CLIR/"

set -a
source "${PROJECT_ROOT}/.env"
set +a

DEVICE=4
DATASET="clef"

while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--device GPU_ID] [--dataset DATASET_NAME]"
      exit 1
      ;;
  esac
done
QRELS_BASE="${DATA_LOCATION}/${DATASET}/qrels"

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
# Create directories if they don't exist
mkdir -p ${RETRIEVAL_BASE}/ranking/${DATASET}/repllama
mkdir -p ${RETRIEVAL_BASE}/results/${DATASET}/repllama
mkdir -p ${RETRIEVAL_BASE}/encodings/${DATASET}/repllama

# Encode docs
for lang in "${LANGUAGES[@]}"; do
  echo "Encoding documents for ${lang}"
  CUDA_VISIBLE_DEVICES=$DEVICE python encode.py \
    --output_dir=temp \
    --model_name_or_path castorini/repllama-v1-7b-lora-passage \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --q_max_len 512 \
    --encode_in_path ${DATA_LOCATION}/${DATASET}/docs/${lang}.jsonl \
    --encoded_save_path ${RETRIEVAL_BASE}/encodings/${DATASET}/repllama/${lang}_docs.pkl
done

# Encode queries
for lang in "${LANGUAGES[@]}"; do
  echo "Encoding queries for ${lang}"
  
  CUDA_VISIBLE_DEVICES=$DEVICE python encode.py \
    --output_dir=temp \
    --model_name_or_path castorini/repllama-v1-7b-lora-passage \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --q_max_len 512 \
    --encode_in_path ${DATA_LOCATION}/${DATASET}/queries/${lang}.jsonl \
    --encoded_save_path ${RETRIEVAL_BASE}/encodings/${DATASET}/repllama/${lang}_queries.pkl \
    --encode_is_qry
done

for pair in "${LANGUAGE_PAIRS[@]}"; do
  SRC=${pair%-*}
  TGT=${pair#*-}
  
  src_full=${LANG_MAP[$SRC]}
  tgt_full=${LANG_MAP[$TGT]}
  
  echo "Processing language pair: $SRC-$TGT"
  
  # Run the retriever
  echo "Running retriever"
  python -m tevatron.faiss_retriever \
    --query_reps "${RETRIEVAL_BASE}/encodings/${DATASET}/repllama/${src_full}_queries.pkl" \
    --passage_reps "${RETRIEVAL_BASE}/encodings/${DATASET}/repllama/${tgt_full}_docs.pkl" \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${RETRIEVAL_BASE}/rankings/${DATASET}/repllama/${src_full}_${tgt_full}.txt
  
  # Convert result to TREC format
  echo "Converting to TREC format"
  python -m tevatron.utils.format.convert_result_to_trec \
    --input ${RETRIEVAL_BASE}/rankings/${DATASET}/repllama/${src_full}_${tgt_full}.txt \
    --output ${RETRIEVAL_BASE}/rankings/${DATASET}/repllama/${src_full}_${tgt_full}.trec \
    --remove_query
  
  rm ${RETRIEVAL_BASE}/rankings/${DATASET}/repllama/${src_full}_${tgt_full}.txt
  
  # Evaluate with trec_eval using target language qrels
  echo "Evaluating using $TGT qrels"
  echo "MAP score:"
  
  ir_measures ${QRELS_BASE}/${tgt_full}.txt \
          ${RETRIEVAL_BASE}/rankings/${DATASET}/repllama/${src_full}_${tgt_full}.trec \
          nDCG@20 AP | tee -a ${RETRIEVAL_BASE}/results/${DATASET}/repllama/${src_full}_${tgt_full}_map.txt
done
