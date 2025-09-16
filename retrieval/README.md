# Retrieval

1. [Models](https://github.com/mainlp/llm-clir/tree/main/retrieval#models) 
2. [Setup](https://github.com/mainlp/llm-clir/tree/main/retrieval#models)
3. [Run bi-encoders](https://github.com/mainlp/llm-clir/tree/main/retrieval#-run-bi-encoders)
   1. [mGTE, M3, E5](https://github.com/mainlp/llm-clir/tree/main/retrieval#-run-bi-encoders)
   2. [NV-Embed-V2](https://github.com/mainlp/llm-clir/tree/main/retrieval#-run-bi-encoders)
   3. [RepLlama](https://github.com/mainlp/llm-clir/tree/main/retrieval#repllama)
4. [BM25](https://github.com/mainlp/llm-clir/tree/main/retrieval#bm25)
5. [Hybrid Rank Fusion](https://github.com/mainlp/llm-clir/tree/main/retrieval#hybrid-rank-fusion-bm25--dense)
6. [Evaluation](https://github.com/mainlp/llm-clir/tree/main/retrieval#-evaluation)

## Models
This folder contains the code to generate the first-stage retrieval results for the following models:

| Model                                                                                       | Huggingface Model ID 🤗                  | Link                                                               | 
|---------------------------------------------------------------------------------------------|------------------------------------------|--------------------------------------------------------------------|
| [**mGTE**](https://aclanthology.org/2024.emnlp-industry.103.pdf) (Zhang et al., EMNLP 2023) | `Alibaba-NLP/gte-multilingual-base`      | [🔗](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)     | 
| [**RepLlama**](https://dl.acm.org/doi/abs/10.1145/3626772.3657951) (Ma el., SIGIR 2024)     | `castorini/repllama-v1-7b-lora-passage`  | [🔗](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) |
| [**M3**](https://aclanthology.org/2024.findings-acl.137/) (Chen et al., ACL 2024)           | `BAAI/bge-m3`                            | [🔗](https://huggingface.co/BAAI/bge-m3)                           |
| [**E5**](https://arxiv.org/pdf/2402.05672) (Wang et al., 2024)                              | `intfloat/multilingual-e5-large`         | [🔗](https://huggingface.co/intfloat/multilingual-e5-large)        | 
| [**NV-Embedd-V2**](https://openreview.net/forum?id=lgsyLSsDRe) (Lee et al., ICLR 2025)      | `nvidia/NV-Embed-v2`                     | [🔗](https://huggingface.co/nvidia/NV-Embed-v2)                    |


## 📦️ Setup

### Environment
The code was tested in the following environment:
  - NVIDIA A100 80GB GPU
  - CUDA V12.9.86
  - conda 25.5.1
  - Python 3.10.15

Due to dependency conflicts, we recommend creating separate environments for retrieval/reranking. Run the following:

```
conda env create -f environment.yml
conda activate encoder
conda install -c conda-forge faiss-gpu
```

The requirements file works for mGTE, M3, and E5. Nv-Embed-V2 and RepLlama require slightly different package versions. We therefore provide extra requirements files for these models.

## 🔍 Run bi-encoders

Below, we show how to run each of the bi-encoders and bm25. Runfiles are written to the `llm-clir/retrieval/rankings` folder.

### mGTE, M3, E5
Results for mGTE, M3, E5 large and NV-Embed-V2 can be generated using the ```encode_and_rank.sh``` script. Usage for this script: 

```
conda env create -f environment.yml --name enc;conda activate enc;conda install -c conda-forge faiss-gpu;yes | pip install -r requirements.txt
./encode_and_rank.sh --device GPU_ID --dataset DATASET --encoder ENCODER_MODEL
```

Valid choices for the --encoder param are: ```mgte, m3, e5, nv```. Valid choices for the --dataset param are: ```clef, ciral```.

### NV-Embed-V2
Nv-Embed-V2 requires different package versions than the other models. Run:
```
conda env create -f environment.yml --name nv;conda activate repllama;conda install -c conda-forge faiss-gpu;yes | pip install -r requirements_nv.txt
./encode_and_rank.sh --device GPU_ID --dataset DATASET --encoder nv
```


### RepLlama
1. Ensure you have access to [Llama-2-7b-hf model](https://huggingface.co/meta-llama/Llama-2-7b-hf) on Huggingface.
2. Reproduce results for RepLlama:
```
chmod +x ./setup_repllama.sh
conda env create -f environment.yml --name repllama;conda activate repllama;conda install -c conda-forge faiss-gpu;./setup_repllama.sh
cd tevatron/examples/repllama
./encode_and_rank.sh --device GPU_ID --dataset DATASET
```
Valid choices for the --dataset param are: ```clef, ciral```.



## BM25
To reproduce the bm25 results for a single language pair, e.g. EN-DE (DT_nllb), run the following command from within `retrieval/` folder:
```
./run_bm25.sh clef en de
```
To run bm25 on all language pairs on both dataset, run `./run_bm25_all.sh`. We used _java-1.21.0-openjdk-amd64_ and *pyserini v.1.2.0*. Note that BM25 (QT) results were taken from [(Adeyemi et al., 2025)](https://aclanthology.org/2024.acl-short.59/), we do not provide query translations for ciral. Running `run_bm25.sh` create an index in `llm-clir/retrieval/indexes/{dataset}/`, and write retrieval results as a trec runfile to `llm-clir/retrieval/rankings/{dataset}/bm25_DT/`.


## Hybrid Rank Fusion (BM25 + Dense)

We also provide the script `hybrid.py` for hybrid rank fusion of BM25 and Dense retrieval results. It reads two TREC run files, combines the ranks using rank reciprocal fusion, and writes the fused results back in TREC format. 

```
python hybrid.py
  --bm25 bm25.trec \
  --dense dense.trec \
  --output hybrid.trec
```

- `--bm25` : Path to BM25 ranking trec file
- `--dense` : Path to Dense ranking trec file
- `--output` : Path to save fused trec file

You can generate trec run files all clef language pairs by running

```
./scripts/run_hybrid.sh clef nv
```

## 💡 Evaluation

To evaluate a single runfile, run: `ir-measures QRELS RUNFILE {MAP,nDCG@20}`
- qrels can be found at `llm-clir/retrieval/rankings/{dataset}/model/{model}.trec`
- runfiles are written to `../data/{dataset}/qrels/{document_language}.txt`
