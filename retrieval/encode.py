import os
import pickle
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.parallel import DataParallel
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
DATA_LOCATION = os.getenv("DATA_LOCATION")
RETRIEVAL_BASE = os.getenv("RETRIEVAL_BASE")

M3_IDENTIFIER = "m3"
MGTE_IDENTIFIER = "mgte"
E5_IDENTIFIER = "e5"
E5_INSTRUCT_IDENTIFIER = "e5_instruct"
NVEMBED_IDENTIFIER = "nv"

def load_field(jsonl_path, field_name):
    df = pd.read_json(jsonl_path, lines=True)
    return df[field_name].values

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode_m3(batch, model, encode_queries):
    if encode_queries:
        embeddings = model.encode_queries(
            batch,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    else:
        embeddings = model.encode_corpus(
            batch,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    return torch.from_numpy(embeddings["dense_vecs"])

def encode_e5(batch, model, tokenizer, device, encode_queries=False):
        # only do this for the e5_instruct model
        if encode_queries:
            task_description = 'Given a web search query, retrieve relevant passages that answer the query'
            batch = [f'Instruct: {task_description}\nQuery: {query}' for query in batch]
        batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict, output_hidden_states=True)
        
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

def encode_mgte(batch, model, tokenizer, device):
        batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict, output_hidden_states=True)
        
        # mgte model dim is 768
        embeddings = outputs.hidden_states[-1][:, 0][:, :768]
        return F.normalize(embeddings, p=2, dim=1)

def encode_nv(batch, model, is_query):
    # taken from: https://huggingface.co/nvidia/NV-Embed-v2
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

    instruction = query_prefix if is_query else ""
    embeddings = model.encode(batch, instruction=instruction, max_length=512)
    return F.normalize(embeddings, p=2, dim=1)

def encode_texts(texts, encoder, model, language, tokenizer = None, device=None, encode_queries=False, batch_size=32):
    all_embeddings = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc=f"Encoding {language} texts"):
        batch = texts[i:i+batch_size].tolist()
        
        if encoder == M3_IDENTIFIER:
            embeddings = encode_m3(batch, model, encode_queries)
        elif encoder == E5_IDENTIFIER:
            embeddings = encode_e5(batch, model, tokenizer, device)
        elif encoder == E5_INSTRUCT_IDENTIFIER:
            embeddings = encode_e5(batch, model, tokenizer, device, encode_queries)
        elif encoder == MGTE_IDENTIFIER:
            embeddings = encode_mgte(batch, model, tokenizer, device)
        elif encoder == NVEMBED_IDENTIFIER:
            embeddings = encode_nv(batch, model, encode_queries)
        else:
            raise ValueError("Invalid encoder selected!")

        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings).cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--encoder", type=str, choices=[M3_IDENTIFIER, E5_IDENTIFIER, E5_INSTRUCT_IDENTIFIER, MGTE_IDENTIFIER, NVEMBED_IDENTIFIER], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encode_queries", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    target = "queries" if args.encode_queries else "docs"
    out_path = os.path.join(RETRIEVAL_BASE, "encodings", args.dataset, args.encoder)
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{args.dataset}_{args.lang}_{target}.pkl")
    if os.path.exists(out_file):
        print(f"Skipping, file exists: {out_file}")
        exit(0)
    
    if args.encoder == M3_IDENTIFIER:
        # load this here, because Nv-Embed does not work when this package is imported
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3',
                        use_fp16=True,
                        pooling_method='cls',
                        devices=device)
        tokenizer = None
    elif args.encoder == E5_IDENTIFIER:
        model_name_or_path = 'intfloat/multilingual-e5-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    elif args.encoder == E5_INSTRUCT_IDENTIFIER:
        model_name_or_path = 'intfloat/multilingual-e5-large-instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    elif args.encoder == MGTE_IDENTIFIER:
        model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    elif args.encoder == NVEMBED_IDENTIFIER:
        model_name_or_path = 'nvidia/NV-Embed-v2'
        tokenizer = None
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        # should not happen as choices are enforced in argparse
        raise ValueError("Invalid encoder selected!")
    
    if args.encoder != M3_IDENTIFIER:
        model = model.to(device)
        model.eval()
      
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = torch.nn.DataParallel(model)
    
    id_field = "qid" if args.encode_queries else "docid"
    jsonl_path = os.path.join(DATA_LOCATION, args.dataset, f"{target}/{args.lang}.jsonl")

    print(f"Loading data from {jsonl_path}")
    df = pd.read_json(jsonl_path, lines=True)
    texts = df["text"].values
    ids = df[id_field].values.tolist()

    embeddings = encode_texts(texts, args.encoder, model, args.lang, tokenizer, device, args.encode_queries, args.batch_size)
    print(f"Saving encodings to {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump((embeddings, ids), f)

if __name__ == "__main__":
    main()