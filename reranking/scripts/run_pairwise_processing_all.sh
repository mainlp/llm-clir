#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${SCRIPT_DIR}/../../.env"

code2full () {
  case "$1" in
    de) echo "german" ;;
    fi) echo "finnish" ;;
    en) echo "english" ;;
    ru) echo "russian" ;;
    it) echo "italian" ;;
    *)  echo "$1" ;;
  esac
}

set -a
source "$ENV_FILE"
set +a

PYTHON_SCRIPT="${SCRIPT_DIR}/../processing_pairwise.py"
DATASETS=("ciral" "clef")

for dataset in "${DATASETS[@]}"; do
  echo "Processing dataset: $dataset"

  base_dir="${RETRIEVAL_BASE}/rankings/${dataset}"
  input_dir="${DATA_LOCATION}/${dataset}"
  
  shopt -s nullglob
  retriever_dirs=( "$base_dir"/*/ )

  for ranking_base in "${retriever_dirs[@]}"; do
    ranking_base="${ranking_base%/}"
    retriever="$(basename "$ranking_base")"

    echo "Retriever: $retriever"

    trec_files=( "$ranking_base"/*.trec )
    docs_dir="${input_dir}/docs"
    qrys_dir="${input_dir}/queries"

    DOC_DIRS=( "${input_dir}/docs" "${input_dir}/docs_translation" )
    SUFFIXES=( "OG" "DT" )

    for trec_file in "${trec_files[@]}"; do
      prefix="${trec_file##*/}"
      prefix="${prefix%.trec}"

      src_code="${prefix:0:2}"
      src_full="$(code2full "$src_code")"

      for i in "${!DOC_DIRS[@]}"; do
        docs_dir="${DOC_DIRS[$i]}"
        suffix="${SUFFIXES[$i]}"

        output_dir="${RERANKING_BASE}/processed_pairwise/${dataset}/${retriever}_${suffix}"
        mkdir -p "$output_dir"

        echo "Running pairwise processing for: $dataset : $retriever : $prefix [${suffix}]"
        
        python "$PYTHON_SCRIPT" \
                  --input_dir "$input_dir" \
                  --ranking_dir "$ranking_base" \
                  --output_dir "$output_dir" \
                  --prefix "$prefix" \
                  --dataset "$dataset" \
                  --translation "$suffix" 
      done
    done
  done
done
