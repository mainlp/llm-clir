#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Environment
# -----------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${SCRIPT_DIR}/../../.env"

set -a
source "$ENV_FILE"
set +a

: "${DATA_LOCATION:?Need DATA_LOCATION in .env}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRANSLATE_PY="${TRANSLATE_PY:-${PROJECT_ROOT}/reranking/nllb_sentence_level.py}"

# -----------------------
# Arguments
# -----------------------
GPU_ID="0"
FORCE=0
DATASETS=(clef ciral)   # default datasets
LANGS=""                # optional: restrict by filename stem (e.g., "german,italian")

usage() {
  cat <<EOF
Usage: $0 [--datasets clef,ciral] [--langs german,italian] [--gpu <id>] [--force]

Description:
  - DOCS input:    \${DATA_LOCATION}/{dataset}/docs/*.jsonl
  - DOCS outputs:  \${DATA_LOCATION}/{dataset}/docs_translated/{src}_{tgt}.jsonl
    * CLEF docs pairs:
        de, fi, it, ru -> en
        it, fi, ru     -> de
        it, ru         -> fi
    * CIRAL docs pairs:
        hausa, somali, swahili, yoruba -> english

  - QUERIES input (CLEF only): \${DATA_LOCATION}/clef/queries/*.jsonl
  - QUERIES outputs (CLEF only): \${DATA_LOCATION}/clef/queries_translated/{src}_{tgt}.jsonl
    * CLEF query pairs:
        en -> it,ru,fi
        de -> it,ru,fi
        fi -> it,ru

Options:
  --datasets   Comma-separated list (default: clef,ciral)
  --langs      Only process these source filename stems (e.g., german,italian)
  --gpu        GPU ID (default: 0)
  --force      Overwrite existing outputs
EOF
  exit 1
}

while (( "$#" )); do
  case "$1" in
    --datasets) IFS=',' read -r -a DATASETS <<< "${2:-}"; shift 2;;
    --langs)    IFS=',' read -r -a LANGS_ARR <<< "${2:-}"; LANGS="${2:-}"; shift 2;;
    --gpu)      GPU_ID="${2:-0}"; shift 2;;
    --force)    FORCE=1; shift 1;;
    -h|--help)  usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

# -----------------------
# Language mappings
# -----------------------
# Map filename language (e.g., 'german') to NLLB code
name2nllb() {
  case "$1" in
    # CLEF
    english)  echo "eng_Latn" ;;
    german)   echo "deu_Latn" ;;
    italian)  echo "ita_Latn" ;;
    finnish)  echo "fin_Latn" ;;
    russian)  echo "rus_Cyrl" ;;
    # CIRAL
    hausa)    echo "hau_Latn" ;;
    somali)   echo "som_Latn" ;;
    swahili)  echo "swh_Latn" ;;
    yoruba)   echo "yor_Latn" ;;
    *)        echo "" ;;
  esac
}

# DOCS: dataset+source -> CSV targets
targets_for_docs() {
  local ds="$1"; local src="$2"
  if [[ "$ds" == "clef" ]]; then
    case "$src" in
      german)  echo "english" ;;                 # de -> en
      finnish) echo "english,german" ;;          # fi -> en,de
      italian) echo "english,german,finnish" ;;  # it -> en,de,fi
      russian) echo "english,german,finnish" ;;  # ru -> en,de,fi
      english) echo "" ;;
      *)       echo "" ;;
    esac
  elif [[ "$ds" == "ciral" ]]; then
    case "$src" in
      hausa|somali|swahili|yoruba) echo "english" ;;
      english) echo "" ;;
      *) echo "" ;;
    esac
  else
    echo ""
  fi
}

# QUERIES (CLEF only): source -> CSV targets
targets_for_queries_clef() {
  local src="$1"
  case "$src" in
    english) echo "italian,russian,finnish" ;;     # en -> it,ru,fi
    german)  echo "italian,russian,finnish" ;;     # de -> it,ru,fi
    finnish) echo "italian,russian" ;;             # fi -> it,ru
    *)       echo "" ;;
  esac
}

# -----------------------
# Translation runner
# -----------------------
# out_subdir is "docs_translated" or "queries_translated"
translate_file() {
  local dataset="$1"
  local src_name="$2"   # e.g., german
  local tgt_name="$3"   # e.g., english
  local in_file="$4"
  local out_subdir="$5" # docs_translated | queries_translated

  local src_nllb; src_nllb="$(name2nllb "$src_name")"
  local tgt_nllb; tgt_nllb="$(name2nllb "$tgt_name")"

  if [[ -z "$src_nllb" || -z "$tgt_nllb" ]]; then
    echo "[skip] Cannot map NLLB codes: $src_name -> $tgt_name ($in_file)"
    return 0
  fi
  if [[ "$src_nllb" == "$tgt_nllb" ]]; then
    echo "[skip] Same language (no-op): $in_file"
    return 0
  fi

  local out_dir="${DATA_LOCATION}/${dataset}/${out_subdir}"
  mkdir -p "$out_dir"
  # Filename style without "to"
  local out_file="${out_dir}/${src_name}_${tgt_name}.jsonl"

  if [[ -f "$out_file" && $FORCE -ne 1 ]]; then
    echo "[skip] Exists: $out_file"
    return 0
  fi

  echo "[run] ($dataset) ${src_name} -> ${tgt_name} : $(basename "$in_file")  -> ${out_subdir}/$(basename "$out_file")"
  "$PYTHON_BIN" "$TRANSLATE_PY" \
    --input_file "$in_file" \
    --output_file "$out_file" \
    --src_lang "$src_nllb" \
    --tgt_lang "$tgt_nllb" \
    --batch_size 256 \
    --max_length 128 \
    --model_name "facebook/nllb-200-1.3B" \
    --gpu "$GPU_ID" \
    --sent_token_lang "$src_nllb"
}

# -----------------------
# Process DOCS for each dataset
# -----------------------
process_docs_for_dataset() {
  local ds="$1"
  local docs_dir="${DATA_LOCATION}/${ds}/docs"

  if [[ ! -d "$docs_dir" ]]; then
    echo "[warn] Missing docs dir: $docs_dir"
    return 0
  fi

  shopt -s nullglob
  local jsonls=( "$docs_dir"/*.jsonl )
  if (( ${#jsonls[@]} == 0 )); then
    echo "[warn] No jsonl files in: $docs_dir"
    return 0
  fi

  echo "---- DOCS: ${ds} ----"
  for f in "${jsonls[@]}"; do
    local base="$(basename "$f")"   # e.g., german.jsonl
    local src_name="${base%.jsonl}" # e.g., german

    # Optional restriction via --langs
    if [[ -n "$LANGS" ]]; then
      local match=0
      for l in "${LANGS_ARR[@]}"; do
        [[ "$l" == "$src_name" ]] && match=1 && break
      done
      [[ $match -eq 1 ]] || { echo "[skip] Not in --langs: $src_name"; continue; }
    fi

    local tgts
    tgts="$(targets_for_docs "$ds" "$src_name")"
    if [[ -z "$tgts" ]]; then
      echo "[skip] No DOCS targets for: $ds/$base"
      continue
    fi

    IFS=',' read -r -a tgt_arr <<< "$tgts"
    for tgt in "${tgt_arr[@]}"; do
      translate_file "$ds" "$src_name" "$tgt" "$f" "docs_translated"
    done
  done
}

# -----------------------
# Process QUERIES (CLEF only)
# -----------------------
process_queries_for_clef() {
  local ds="clef"
  local q_dir="${DATA_LOCATION}/${ds}/queries"

  if [[ ! -d "$q_dir" ]]; then
    echo "[warn] Missing queries dir: $q_dir"
    return 0
  fi

  shopt -s nullglob
  local jsonls=( "$q_dir"/*.jsonl )
  if (( ${#jsonls[@]} == 0 )); then
    echo "[warn] No jsonl files in: $q_dir"
    return 0
  fi

  echo "---- QUERIES: ${ds} ----"
  for f in "${jsonls[@]}"; do
    local base="$(basename "$f")"   # e.g., english.jsonl
    local src_name="${base%.jsonl}" # e.g., english

    # Optional restriction via --langs (reuses same filter)
    if [[ -n "$LANGS" ]]; then
      local match=0
      for l in "${LANGS_ARR[@]}"; do
        [[ "$l" == "$src_name" ]] && match=1 && break
      done
      [[ $match -eq 1 ]] || { echo "[skip] Not in --langs: $src_name"; continue; }
    fi

    local tgts
    tgts="$(targets_for_queries_clef "$src_name")"
    if [[ -z "$tgts" ]]; then
      echo "[skip] No QUERIES targets for: $ds/$base"
      continue
    fi

    IFS=',' read -r -a tgt_arr <<< "$tgts"
    for tgt in "${tgt_arr[@]}"; do
      translate_file "$ds" "$src_name" "$tgt" "$f" "queries_translated"
    done
  done
}

# -----------------------
# Main
# -----------------------
for ds in "${DATASETS[@]}"; do
  echo "======== DATASET: $ds ========"
  process_docs_for_dataset "$ds"

  # Only CLEF has query translations per your requirement
  if [[ "$ds" == "clef" ]]; then
    process_queries_for_clef
  fi
done

echo "Done."
