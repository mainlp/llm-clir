# Reranking

1. [Models](https://github.com/mainlp/llm-clir/tree/main/reranking#models)
2. [Setup](https://github.com/mainlp/llm-clir/tree/main/reranking#%EF%B8%8F-setup)
3. [Listwise Reranking: RankZephyr](https://github.com/mainlp/llm-clir/tree/main/reranking#-listwise-reranking-rankzephyr)
   1. [Input Format](https://github.com/mainlp/llm-clir/tree/main/reranking#input-format)
   2. [Usage](https://github.com/mainlp/llm-clir/tree/main/reranking#usage)
4. [Listwise Reranking: RankGPT](https://github.com/mainlp/llm-clir/tree/main/reranking#-listwise-reranking-rankgpt)
   1. [Input Format](https://github.com/mainlp/llm-clir/tree/main/reranking#input-format-1)
   2. [Usage](https://github.com/mainlp/llm-clir/tree/main/reranking#usage-1)
5. [Pairwise Reranking](https://github.com/mainlp/llm-clir/tree/main/reranking#%EF%B8%8F-pairwise-reranking)
   1. [Input Format](https://github.com/mainlp/llm-clir/tree/main/reranking#input-format-2)
   2. [Usage](https://github.com/mainlp/llm-clir/tree/main/reranking#usage-2)
6. [Results](https://github.com/mainlp/llm-clir/tree/main/reranking#results)
7. [Sentence-Level Translation with NLLB](https://github.com/mainlp/llm-clir/tree/main/reranking#-sentence-level-translation-with-nllb)
   1. [Input Format](https://github.com/mainlp/llm-clir/tree/main/reranking#input-format-3)
   2. [Usage](https://github.com/mainlp/llm-clir/tree/main/reranking#usage-3)

## Models

This folder contains the code to generate the second-stage reranker results for the following models:

| Model                                                                                                | Reranking | Open AI / Huggingface Model ID 🤗       | Link                                                          |
|------------------------------------------------------------------------------------------------------|-----------|-----------------------------------------|---------------------------------------------------------------|
| [**RankZephyr**](https://arxiv.org/abs/2312.02724) (Pradeep et al., 2023)                            | listwise  | `castorini/rank_zephyr_7b_v1_full`      | [🔗](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) |
| [**RankGPT<sub>3.5</sub>**](https://aclanthology.org/2023.emnlp-main.923/) (Sun et al., EMNLP 2023)  | listwise  | `gpt-3.5-turbo`                         | [🔗](https://openai.com/index/chatgpt/)                       |
| [**RankGPT<sub>4.1</sub>**](https://aclanthology.org/2023.emnlp-main.923/) (Sun et al., EMNLP 2023)  | listwise  | `gpt-4.1`                               | [🔗](https://openai.com/index/gpt-4-1/)                       |
| [**Aya-101**](https://arxiv.org/abs/2402.07827) (Üstün et al., 2024)                                 | pairwise  | `CohereLabs/aya-101`                    | [🔗](https://huggingface.co/CohereLabs/aya-101)               |
| [**Llama-3.1-8B-Instruct**](https://arxiv.org/abs/2407.21783) (Grattafiori et al., 2024)             | pairwise  | `meta-llama/Meta-Llama-3.1-8B-Instruct` | [🔗](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |


## 📦️ Setup
We recommend creating a separate Conda environment for reranking.
You can create and activate a new environment named `reranking` by running:

```
conda env create -f reranking_env.yml
conda activate reranking
```

## 🍃 Listwise Reranking: RankZephyr

### Input Format
The `rankzephyr.py` script expects `.jsonl` files as input. Each line in the file should be a JSON object with the following format:

```
{
    "query": 
    {
      "qid": "1", 
      "text": "What is the capital of France?"
      }, 
        
    "candidates": [
        {
          "docid": "D1", 
          "score": 0.89, 
          "doc": {"content": "Paris is the capital and most populous city of France."}
          },
        {
          "docid": "D2", 
          "score": 0.59, 
          "doc": {"content": "France is a country in Western Europe"}
          },
        {
          "docid": "D3", 
          "score": 0.2, 
          "doc": {"content": "Berlin is the capital of Germany."}
          }
        ]
}
```
  - `qid`: Query ID
  - `text`: A string representing the user's query
  - `candidates`: A list of document objects, each with a `docid`, `score` and `doc` field

<!-- **Batch Preprocessing for All Languages** -->

To process the data into the format required for listwise second-stage reranking, run

```
bash run_listwise_processing_all.sh
```

The script reads queries, documents and input rankings (trec run files), and prepares input reranking files:

```
processed_listwise/clef/
└── {e5,mgte,m3,repllama,nv}_{OG,DT}
          ├── english_finnish.jsonl
          ├── ...
          └── german_russian.jsonl
```
and
```
processed_listwise/ciral/
└── {e5,mgte,m3,repllama,nv}_{OG,DT}
          ├── hausa.jsonl
          ├── somali.jsonl
          ├── swahili.jsonl
          └── yoruba.jsonl
```
Folders ending with `DT` contain input rankings with documents translated to the query language, folders ending with `OG` contain input rankings with documents written in their original document language. 

### Usage
Run the `rankzephyr.py` script as follows:
```
CUDA_VISIBLE_DEVICES=0 python rankzephyr.py \
    --dataset <clef|ciral> \
    --retriever <retriever_name>
```
  - `--dataset`: Dataset to evaluate, either `clef` or `ciral`
  - `--retriever`: Subdirectory name under processed_listwise/ or processed_pairwise/. For example, BM25_OG or m3_DT.
  
## 🤖 Listwise Reranking: RankGPT

### Input Format

RankGPT framework requires the same input format as RankZephyr: each `.jsonl` file should contain ranking requests in standard format.

### Usage
First set the OpenAI API key in `.env`, then run the `rankgpt.py` script as follows:
```
CUDA_VISIBLE_DEVICES=0 python rankgpt.py \
    --dataset <clef|ciral> \
    --retriever <retriever_name> \
    --model <model_name>
```
  - `--dataset`: Dataset name, either `clef` or `ciral`
  - `--retriever`: Subdirectory name under processed_listwise/ or processed_pairwise/. For example, BM25_OG or m3_DT.
  - `--model`: GPT model name (default: gpt-4.1)



## ⚖️ Pairwise Reranking

### Input Format
Pairwise reranking scripts expect each input `.tsv` file to follow this format:
```
query_id<TAB>doc_id<TAB>score<TAB>doc_content
```

  - `query_id`: Query ID
  - `doc_id`: Docuemnt ID
  - `score`: Score from the first-stage retriever
  - `doc_content`: Full text of the corresponding document


To create the input files in this format, run the following command:

```
bash run_pairwise_processing_all.sh
```

### Usage

1. Login with huggingface CLI `hf auth login` (old: `huggingface-cli login`) and enter your `$HF_TOKEN`.
2. Run pairwise reranking with a specified model:

```
bash run_pairwise_all.sh \
  --gpu 0 \
  --model <model_name> \
  --dataset <dataset> \
  --passage_length 300 \
  --retriever <retriever>
```

 - `--gpu`	Set CUDA_VISIBLE_DEVICES
 - `--model`	Model name (Llama-3.1-8B-Instruct or Aya-101)
 - `--dataset`: Dataset name, either `clef` or `ciral` (default: `clef`)
 - `--passage_length` Max token length for each passage (default: 300)
 - `--retriever` run reranking on all files (=language pairs) in folder `processed_pairwise/<dataset>/<retriever>`

Example:
```
./run_pairwise_all.sh --gpu 1 --model Llama-3.1-8B-Instruct --dataset clef --retriever nv_OG
```

## Results

The reranking run files are saved under `/reranking/rankings`. Each result file is in TREC format (.trec) and contains lines structured as follows:

```
query_id Q0 doc_id ranking score model_identifier
```

You can evaluate these outputs using standard TREC evaluation tools such as  `trec_eval` or `pytrec_eval`.


## 🌎 Sentence-Level Translation with NLLB

This module provides a script to perform sentence-level multilingual translation using Meta's [NLLB-200](https://huggingface.co/facebook/nllb-200-1.3B) model (`facebook/nllb-200-1.3B`). Code and bash script are provided under the `/reranking/nllb` directory. 

### Input Format
Input should be tab-separated file (`.txt` or `.tsv`) with at least 2 columns:
```
id<TAB>text
```

### Usage
  - Option 1: Python CLI

```
CUDA_VISIBLE_DEVICES=0 python nllb_sentence_translate.py \
  --input_file example_data/de_queries_desc.txt \
  --output_file output/nllb_de2en.tsv \
  --src_lang deu_Latn \
  --tgt_lang eng_Latn \
  --sent_token_lang german
```

  - Option 2: Bash Script

```
bash run_translate.sh 0 deu_Latn eng_Latn example_data/de_queries_desc.txt output/nllb_de2en.tsv german
```

### Supported Languages
Use ISO 639-3 codes in NLLB token format, e.g.:
 - German: `deu_Latn`
 - Russian: `rus_Cyrl`
 - English: `eng_Latn`
 - Italian: `ita_Latn`
 